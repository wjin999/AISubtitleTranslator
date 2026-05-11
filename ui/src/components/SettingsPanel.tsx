import type { ModalType } from "./PromptModal";

interface Props {
  apiKey: string; setApiKey: (v: string) => void;
  sumModel: string; setSumModel: (v: string) => void;
  transModel: string; setTransModel: (v: string) => void;
  savePath: string; setSavePath: (v: string) => void;
  concurrency: number; setConcurrency: (v: number) => void;
  maxOutputTokens: number; setMaxOutputTokens: (v: number) => void;
  requestTimeout: number; setRequestTimeout: (v: number) => void;
  sourceLanguage: string; setSourceLanguage: (v: string) => void;
  mergeEnabled: boolean; setMergeEnabled: (v: boolean) => void;
  saveMergedSubtitles: boolean; setSaveMergedSubtitles: (v: boolean) => void;
  qualityCheckEnabled: boolean; setQualityCheckEnabled: (v: boolean) => void;
  setActiveModal: (m: ModalType) => void;
  resetToDefaults: () => void;
}

interface NumberFieldProps {
  label: string;
  value: number;
  setValue: (v: number) => void;
  min: number;
  max: number;
  step?: number;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function NumberField({ label, value, setValue, min, max, step = 1 }: NumberFieldProps) {
  const precision = step.toString().includes(".") ? step.toString().split(".")[1].length : 0;
  const apply = (next: number) => {
    const clamped = clamp(next, min, max);
    setValue(Number(clamped.toFixed(precision)));
  };

  return (
    <div className="field-group number-field">
      <label>{label}</label>
      <div className="number-control">
        <input
          type="number"
          value={value}
          onChange={(e) => apply(parseFloat(e.target.value) || min)}
          min={min}
          max={max}
          step={step}
        />
        <div className="number-stepper">
          <button type="button" onClick={() => apply(value + step)}>+</button>
          <button type="button" onClick={() => apply(value - step)}>-</button>
        </div>
      </div>
    </div>
  );
}

export default function SettingsPanel(props: Props) {
  const {
    apiKey, setApiKey,
    sumModel, setSumModel, transModel, setTransModel,
    savePath, setSavePath, concurrency, setConcurrency,
    maxOutputTokens, setMaxOutputTokens,
    requestTimeout, setRequestTimeout,
    sourceLanguage, setSourceLanguage, mergeEnabled, setMergeEnabled,
    saveMergedSubtitles, setSaveMergedSubtitles,
    qualityCheckEnabled, setQualityCheckEnabled,
    setActiveModal, resetToDefaults,
  } = props;

  return (
    <div className="view-content scrollable">
      <div className="field-group">
        <label>保存路径</label>
        <input type="text" value={savePath} onChange={(e) => setSavePath(e.target.value)} placeholder="留空则保存到桌面" />
      </div>
      <div className="ti8-divider" />
      <div className="field-group">
        <label>DeepSeek API KEY</label>
        <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="留空则读取 DEEPSEEK_API_KEY" />
      </div>
      <div className="ti8-divider" />
      <div className="field-group">
        <label>自定义术语表</label>
        <button className="ti8-prompt-btn" style={{ borderColor: '#d4af37', color: '#d4af37' }} onClick={() => setActiveModal("glossary")}>
          [ 编辑术语表 ]
        </button>
      </div>
      <div className="ti8-divider" />
      <div className="field-group">
        <label>源语言</label>
        <select value={sourceLanguage} onChange={(e) => setSourceLanguage(e.target.value)}>
          <option value="en">英语</option>
          <option value="ja">日语</option>
          <option value="ko">韩语</option>
        </select>
      </div>
      <div className="field-group checkbox-field">
        <label>字幕合并</label>
        <label className="checkbox-row">
          <input type="checkbox" checked={mergeEnabled} onChange={(e) => setMergeEnabled(e.target.checked)} />
          <span>启用 spaCy 智能合并</span>
        </label>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={saveMergedSubtitles}
            disabled={!mergeEnabled}
            onChange={(e) => setSaveMergedSubtitles(e.target.checked)}
          />
          <span>保存 spaCy 合并后的字幕文件</span>
        </label>
      </div>
      <div className="field-group checkbox-field">
        <label>自动质检</label>
        <label className="checkbox-row">
          <input type="checkbox" checked={qualityCheckEnabled} onChange={(e) => setQualityCheckEnabled(e.target.checked)} />
          <span>翻译后自动运行质量检查</span>
        </label>
      </div>
      <div className="ti8-divider" />
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
      <div className="ti8-divider" />
      <NumberField label="并发数" value={concurrency} setValue={setConcurrency} min={1} max={50} />
      <NumberField label="最大输出 Tokens" value={maxOutputTokens} setValue={setMaxOutputTokens} min={256} max={32768} step={256} />
      <NumberField label="请求超时（秒）" value={requestTimeout} setValue={setRequestTimeout} min={5} max={600} />
      <div className="ti8-divider" />
      <button className="ti8-reset-btn" onClick={resetToDefaults}>恢复默认设置</button>
    </div>
  );
}
