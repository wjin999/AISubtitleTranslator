import { useState, useEffect, useRef } from "react";
import "./App.css";

const DEF_SUM_PROMPT = `你是一名专业的内容分析师。请阅读以下文本，生成一份简洁的背景摘要，内容包括：主要话题、关键术语、角色关系以及整体基调。用中文输出，不超过 150 字。`;
const DEF_TRANS_PROMPT = `你是一名专业的影视字幕翻译专家，负责将英文字幕翻译成简体中文。

## 核心要求：
1. 输出合法 JSON 格式：{"translations": [{"id": 0, "text": "翻译"}, ...]}
2. 输出的条目数量必须与输入完全一致
3. 每条译文必须简洁（中文约 3-5 字符对应一秒屏幕时间）

## 翻译最佳实践：
4. 使用自然、口语化的中文，适合对话场景
5. 保留说话者的语气和情感（愤怒、低语、讽刺、兴奋等）
6. 保持角色语气在全篇字幕中一致
7. 遇到习语、双关语或文化特定内容时，采用意译而非直译

## 字幕标点规范（必须严格遵守）：
8. 句末不加句号（。）：无论是陈述句还是祈使句，字幕结尾一律不写句号（。）。字幕的出现和消失本身就起到了断句的作用。
9. 必须保留问号（？）和叹号（！）：用于准确传达疑问或强烈语气。
10. 句中停顿用空格代替逗号（，）：句子内部的停顿使用空格（半角或全角均可），不使用逗号或顿号。
11. 省略号（……）和破折号（——）：表示话语未说完、被打断或声音拖长时规范使用。
12. 引号（""）和括号（（））：专有名词、画外音或内心独白时正常使用。

## JSON 格式示例：
{"translations": [{"id": 0, "text": "你好"}, {"id": 1, "text": "世界"}]}`;

const DEF_URL = "https://api.deepseek.com";
const DEF_SUM_MODEL = "deepseek-v4-pro";
const DEF_TRANS_MODEL = "deepseek-v4-pro";

type LogEntry = { text: string; isError: boolean };
type ModalType = "sum" | "trans" | "glossary" | null;

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [activeModal, setActiveModal] = useState<ModalType>(null);
  
  const [isWorking, setIsWorking] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("请选择字幕文件");
  const [isError, setIsError] = useState(false);
  
  const [logs, setLogs] = useState<LogEntry[]>([{text: "- 等待操作...", isError: false}]);
  const logsContainerRef = useRef<HTMLDivElement>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const [apiKey, setApiKey] = useState(() => localStorage.getItem("apiKey") || "");
  const [url, setUrl] = useState(() => localStorage.getItem("url") || DEF_URL);
  const [sumModel, setSumModel] = useState(() => localStorage.getItem("sumModel") || DEF_SUM_MODEL);
  const [transModel, setTransModel] = useState(() => localStorage.getItem("transModel") || DEF_TRANS_MODEL);
  const [sumPrompt, setSumPrompt] = useState(() => localStorage.getItem("sumPrompt") || DEF_SUM_PROMPT);
  const [transPrompt, setTransPrompt] = useState(() => localStorage.getItem("transPrompt") || DEF_TRANS_PROMPT);
  const [savePath, setSavePath] = useState(() => localStorage.getItem("savePath") || "");
  const [glossary, setGlossary] = useState(() => localStorage.getItem("glossary") || "");

  useEffect(() => {
    document.title = "wjin999/AISubtitleTranslator";
  }, []);

  useEffect(() => {
    const container = logsContainerRef.current;
    if (container && logsEndRef.current) {
      const isScrolledToBottom = container.scrollHeight - container.clientHeight <= container.scrollTop + 1;
      if (isScrolledToBottom) {
        logsEndRef.current.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [logs]);

  useEffect(() => {
    localStorage.setItem("apiKey", apiKey);
    localStorage.setItem("url", url);
    localStorage.setItem("sumModel", sumModel);
    localStorage.setItem("transModel", transModel);
    localStorage.setItem("sumPrompt", sumPrompt);
    localStorage.setItem("transPrompt", transPrompt);
    localStorage.setItem("savePath", savePath);
    localStorage.setItem("glossary", glossary);
  }, [apiKey, url, sumModel, transModel, sumPrompt, transPrompt, savePath, glossary]);

  const resetToDefaults = () => {
    if (window.confirm("确认要恢复到系统默认配置吗？")) {
      setApiKey(""); setUrl(DEF_URL); setSumModel(DEF_SUM_MODEL); setTransModel(DEF_TRANS_MODEL);
      setSumPrompt(DEF_SUM_PROMPT); setTransPrompt(DEF_TRANS_PROMPT); setSavePath(""); setGlossary("");
      setLogs([{text: "- 配置已重置", isError: false}]);
    }
  };

  const run = async () => {
    if (!file) return;
    setIsWorking(true); setProgress(0); setIsError(false);
    setStatus("正在启动后台服务...");
    setLogs([{text: "- 正在启动后台服务，请稍候...", isError: false}]);

    // 等待后端服务就绪（最多重试 15 秒）
    let backendReady = false;
    for (let i = 0; i < 30; i++) {  // 30 次 × 500ms = 15 秒
      try {
        const healthRes = await fetch("http://127.0.0.1:8000/api/health", { method: "GET" });
        if (healthRes.ok) {
          backendReady = true;
          break;
        }
      } catch (_) {
        // 服务尚未就绪，继续等待
      }
      await new Promise(r => setTimeout(r, 500));
    }

    if (!backendReady) {
      setIsWorking(false); setIsError(true);
      setLogs(prev => [...prev, {text: "[错误] 无法连接到本地后台服务，请确认没有其他程序占用 8000 端口", isError: true}]);
      setStatus("连接失败");
      return;
    }

    setStatus("正在准备翻译...");
    setLogs(prev => [...prev, {text: "- 后台服务就绪，开始提交翻译...", isError: false}]);

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

    try {
      const res = await fetch("http://127.0.0.1:8000/api/translate", { method: "POST", body: form });
      const data = await res.json();
      
      if (data.status === "success" && data.job_id) {
        const jobId = data.job_id;
        
        const timer = setInterval(async () => {
          try {
            const statusRes = await fetch(`http://127.0.0.1:8000/api/status/${jobId}`);
            const statusData = await statusRes.json();

            if (statusData.logs && Array.isArray(statusData.logs)) {
              setLogs(statusData.logs);
            }

            if (statusData.status === "error") {
              clearInterval(timer); setIsWorking(false); setIsError(true);
              setStatus("翻译失败，请查看下方日志");
            } else if (statusData.status === "completed") {
              clearInterval(timer); setProgress(100); setIsWorking(false);
              setStatus("翻译完成！");
            } else {
              setProgress(statusData.progress || 0);
              setStatus(`正在翻译... ${statusData.progress || 0}%`);
            }
          } catch(err) {
            clearInterval(timer); setIsWorking(false); setIsError(true);
            setLogs(prev => [...prev, {text: "[错误] 与后台服务的连接断开", isError: true}]);
            setStatus("连接断开");
          }
        }, 1000);
      }
    } catch (e) {
      setIsWorking(false); setIsError(true);
      setLogs(prev => [...prev, {text: "[错误] 无法连接到本地后台服务", isError: true}]);
      setStatus("连接失败");
    }
  };

  const getModalTitle = () => {
    if (activeModal === "sum") return "概括提示词";
    if (activeModal === "trans") return "翻译提示词";
    return "自定义术语表";
  };

  const getModalValue = () => {
    if (activeModal === "sum") return sumPrompt;
    if (activeModal === "trans") return transPrompt;
    return glossary;
  };

  const setModalValue = (val: string) => {
    if (activeModal === "sum") setSumPrompt(val);
    else if (activeModal === "trans") setTransPrompt(val);
    else if (activeModal === "glossary") setGlossary(val);
  };

  return (
    <div className="ti8-container">
      <div className="terminal-card">
        
        <div className="card-header">
          <span className="ti8-emblem" style={{ fontWeight: 'bold', letterSpacing: '1px' }}>
            wjin999/AISubtitleTranslator
          </span>
          <button className="settings-toggle" onClick={() => setShowSettings(!showSettings)}>
            {showSettings ? "[ 返回终端 ]" : "[ 系统配置 ]"}
          </button>
        </div>

        {!showSettings ? (
          <div className="view-content">
            <div className="ti8-upload">
              <input type="file" accept=".srt" onChange={(e) => {
                if(e.target.files?.[0]) { 
                  setFile(e.target.files[0]); 
                  setLogs([{text: `- 已选择文件: ${e.target.files[0].name}`, isError: false}]);
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

            {(isWorking || progress === 100) && !isError && (
              <div className="progress-container">
                <div className="progress-bar" style={{ width: `${progress}%` }}></div>
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
        ) : (
          <div className="view-content scrollable">
            <div className="field-group">
              <label>保存路径</label>
              <input type="text" value={savePath} onChange={(e) => setSavePath(e.target.value)} placeholder="留空则保存在程序所在的文件夹" />
            </div>
            <div className="ti8-divider"></div>
            <div className="field-group">
              <label>API KEY</label>
              <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="留空则读取系统环境变量" />
            </div>
            <div className="field-group">
              <label>接口地址 (base_url)</label>
              <input type="text" value={url} onChange={(e) => setUrl(e.target.value)} />
            </div>
            <div className="ti8-divider"></div>
            <div className="field-group">
              <label>自定义术语表</label>
              <button className="ti8-prompt-btn" style={{ borderColor: '#d4af37', color: '#d4af37' }} onClick={() => setActiveModal("glossary")}>
                [ 编辑术语表 ]
              </button>
            </div>
            <div className="ti8-divider"></div>
            <div className="field-group">
              <label>概括模型</label>
              <input type="text" value={sumModel} onChange={(e) => setSumModel(e.target.value)} />
              <button className="ti8-prompt-btn" onClick={() => setActiveModal("sum")}>[ 编辑概括提示词 ]</button>
            </div>
            <div className="field-group">
              <label>翻译模型</label>
              <input type="text" value={transModel} onChange={(e) => setTransModel(e.target.value)} />
              <button className="ti8-prompt-btn" onClick={() => setActiveModal("trans")}>[ 编辑翻译提示词 ]</button>
            </div>
            <button className="ti8-reset-btn" onClick={resetToDefaults}>恢复默认设置</button>
          </div>
        )}
      </div>

      {activeModal && (
        <div className="ti8-overlay">
          <div className="ti8-modal">
            <div className="modal-top">
              <span>{getModalTitle()}</span>
              <button onClick={() => setActiveModal(null)}>确认保存</button>
            </div>
            
            {/* 新增：仅在编辑术语表时显示说明文本 */}
            {activeModal === "glossary" && (
              <div style={{ fontSize: '12px', color: '#5c8577', marginBottom: '10px', lineHeight: '1.4' }}>
                * 此内容直接提供给大模型作为翻译参考。<br/>
                * 建议格式：原文: 译文（冒号中英皆可），每行一个。<br/>
                * 示例：<br/>
                  &nbsp;&nbsp;Cyberpunk: 赛博朋克<br/>
                  &nbsp;&nbsp;Night City: 夜之城
              </div>
            )}

            <textarea 
              value={getModalValue()} 
              onChange={(e) => setModalValue(e.target.value)} 
              // 动态调整高度，让带有说明文本的弹窗布局依然协调
              style={{ height: activeModal === "glossary" ? '220px' : '320px' }}
              spellCheck={false} 
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;