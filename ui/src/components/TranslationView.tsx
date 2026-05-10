import { useEffect, useRef, type Dispatch, type SetStateAction } from "react";

export const BACKEND_HOST = "127.0.0.1";
export const BACKEND_PORT = "18770";
export const BACKEND_BASE = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
const POLL_INTERVAL = 2500;
const BACKEND_READY_ATTEMPTS = 120;

type LogEntry = { text: string; isError: boolean };

interface Props {
  file: File | null;
  setFile: (f: File | null) => void;
  isWorking: boolean;
  setIsWorking: (v: boolean) => void;
  progress: number;
  setProgress: (v: number) => void;
  status: string;
  setStatus: (v: string) => void;
  isError: boolean;
  setIsError: (v: boolean) => void;
  logs: LogEntry[];
  setLogs: Dispatch<SetStateAction<LogEntry[]>>;
  apiKey: string;
  url: string;
  sumModel: string;
  transModel: string;
  sumPrompt: string;
  transPrompt: string;
  savePath: string;
  glossary: string;
  concurrency: number;
  sourceLanguage: string;
  mergeEnabled: boolean;
}

export default function TranslationView(props: Props) {
  const {
    file, setFile, isWorking, setIsWorking, progress, setProgress,
    status, setStatus, isError, setIsError, logs, setLogs,
    apiKey, url, sumModel, transModel,
    sumPrompt, transPrompt, savePath, glossary, concurrency,
    sourceLanguage, mergeEnabled,
  } = props;

  const logsContainerRef = useRef<HTMLDivElement>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const jobIdRef = useRef<string | null>(null);
  const pollingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = () => {
    if (pollingTimerRef.current !== null) {
      clearInterval(pollingTimerRef.current);
      pollingTimerRef.current = null;
    }
  };

  useEffect(() => {
    const container = logsContainerRef.current;
    if (container && logsEndRef.current) {
      const atBottom = container.scrollHeight - container.clientHeight <= container.scrollTop + 1;
      if (atBottom) {
        logsEndRef.current.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [logs]);

  const cancelTranslation = async () => {
    if (!jobIdRef.current) return;
    stopPolling();
    try {
      await fetch(`${BACKEND_BASE}/api/cancel/${jobIdRef.current}`, { method: "POST" });
    } catch { /* ignore */ }
    setIsWorking(false);
    setStatus("已取消");
    setLogs(prev => [...prev, { text: "- 任务已取消", isError: false }]);
    jobIdRef.current = null;
  };

  const run = async () => {
    if (!file) return;
    stopPolling();
    setIsWorking(true);
    setProgress(0);
    setIsError(false);
    setStatus("正在启动后台服务...");
    setLogs([{ text: "- 正在启动后台服务，请稍候...", isError: false }]);

    let backendReady = false;
    for (let i = 0; i < BACKEND_READY_ATTEMPTS; i++) {
      try {
        const healthRes = await fetch(`${BACKEND_BASE}/api/health`, { method: "GET" });
        if (healthRes.ok) { backendReady = true; break; }
      } catch { /* wait */ }
      await new Promise(r => setTimeout(r, 500));
    }

    if (!backendReady) {
      setIsWorking(false);
      setIsError(true);
      setLogs(prev => [...prev, { text: `[错误] 无法连接到本地后台服务，请确认没有其他程序占用 ${BACKEND_PORT} 端口`, isError: true }]);
      setStatus("连接失败");
      return;
    }

    setStatus("正在准备翻译...");
    setLogs(prev => [...prev, { text: "- 后台服务就绪，开始提交翻译...", isError: false }]);

    const form = new FormData();
    form.append("file", file);
    form.append("api_key", apiKey);
    form.append("base_url", url);
    form.append("summary_model_name", sumModel);
    form.append("model_name", transModel);
    form.append("summary_prompt", sumPrompt);
    form.append("translation_prompt", transPrompt);
    form.append("save_path", savePath);
    form.append("glossary", glossary);
    form.append("concurrency", String(concurrency));
    form.append("source_language", sourceLanguage);
    form.append("merge_enabled", String(mergeEnabled));

    try {
      const res = await fetch(`${BACKEND_BASE}/api/translate`, { method: "POST", body: form });
      const data = await res.json();

      if (!data.job_id) {
        setIsWorking(false);
        setIsError(true);
        setStatus(data.error || "翻译请求失败");
        setLogs(prev => [...prev, { text: `[错误] ${data.error || "翻译请求失败"}`, isError: true }]);
        return;
      }

      jobIdRef.current = data.job_id;

      pollingTimerRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(`${BACKEND_BASE}/api/status/${jobIdRef.current}`);
          const statusData = await statusRes.json();

          if (statusData.logs && Array.isArray(statusData.logs)) {
            setLogs(statusData.logs);
          }

          switch (statusData.status) {
            case "error":
              stopPolling();
              setIsWorking(false);
              setIsError(true);
              setStatus("翻译失败，请查看下方日志");
              break;
            case "completed":
              stopPolling();
              setProgress(100);
              setIsWorking(false);
              setStatus("翻译完成！");
              break;
            case "cancelled":
              stopPolling();
              setIsWorking(false);
              setStatus("已取消");
              break;
            default:
              setProgress(statusData.progress || 0);
              setStatus(`正在翻译... ${statusData.progress || 0}%`);
          }
        } catch {
          stopPolling();
          setIsWorking(false);
          setIsError(true);
          setLogs(prev => [...prev, { text: "[错误] 与后台服务的连接断开", isError: true }]);
          setStatus("连接断开");
        }
      }, POLL_INTERVAL);
    } catch {
      setIsWorking(false);
      setIsError(true);
      setLogs(prev => [...prev, { text: "[错误] 无法连接到本地后台服务", isError: true }]);
      setStatus("连接失败");
    }
  };

  return (
    <div className="view-content">
      <div className="ti8-upload">
        <input type="file" accept=".srt" onChange={(e) => {
          if (e.target.files?.[0]) {
            setFile(e.target.files[0]);
            setLogs([{ text: `- 已选择文件: ${e.target.files[0].name}`, isError: false }]);
            setStatus("文件已就绪");
            setProgress(0); setIsWorking(false); setIsError(false);
          }
        }} id="ti-up" />
        <label htmlFor="ti-up" className="ti8-btn-upload">
          <span className="label-main">{file ? file.name : "点击选择 SRT 字幕文件"}</span>
        </label>
      </div>

      <button className={`ti8-execute ${file ? "ready" : ""}`} onClick={run} disabled={!file || isWorking}>
        {isWorking ? "正在翻译中" : "开始翻译"}
      </button>

      {isWorking && (
        <button className="ti8-cancel-btn" onClick={cancelTranslation}>
          取消翻译
        </button>
      )}

      {(isWorking || progress === 100) && !isError && (
        <div className="progress-container">
          <div className="progress-bar" style={{ width: `${progress}%` }} />
        </div>
      )}

      <div className={`ti8-status ${isError ? "is-error" : ""}`}>{status}</div>

      <div className="terminal-logs-container" ref={logsContainerRef}>
        {(logs || []).map((log, index) => (
          <div key={index} className={`log-line ${log.isError ? "error" : ""}`}>
            {log.text}
          </div>
        ))}
        <div ref={logsEndRef} />
      </div>
    </div>
  );
}
