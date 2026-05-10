import type { ModalType } from "./PromptModal";

interface Props {
  apiKey: string; setApiKey: (v: string) => void;
  url: string; setUrl: (v: string) => void;
  sumModel: string; setSumModel: (v: string) => void;
  transModel: string; setTransModel: (v: string) => void;
  savePath: string; setSavePath: (v: string) => void;
  concurrency: number; setConcurrency: (v: number) => void;
  sourceLanguage: string; setSourceLanguage: (v: string) => void;
  mergeEnabled: boolean; setMergeEnabled: (v: boolean) => void;
  setActiveModal: (m: ModalType) => void;
  resetToDefaults: () => void;
}

export default function SettingsPanel(props: Props) {
  const {
    apiKey, setApiKey, url, setUrl,
    sumModel, setSumModel, transModel, setTransModel,
    savePath, setSavePath, concurrency, setConcurrency,
    sourceLanguage, setSourceLanguage, mergeEnabled, setMergeEnabled,
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
        <label>API KEY</label>
        <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="留空则读取系统环境变量" />
      </div>
      <div className="field-group">
        <label>接口地址 (base_url)</label>
        <input type="text" value={url} onChange={(e) => setUrl(e.target.value)} />
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
      <div className="field-group">
        <label>并发数</label>
        <input type="number" value={concurrency} onChange={(e) => setConcurrency(parseInt(e.target.value) || 8)} min={1} max={50} />
      </div>
      <div className="ti8-divider" />
      <button className="ti8-reset-btn" onClick={resetToDefaults}>恢复默认设置</button>
    </div>
  );
}
