import { useState, useEffect, useRef } from "react";

export const DEF_URL = "https://api.deepseek.com";
export const DEF_SUM_MODEL = "deepseek-v4-pro";
export const DEF_TRANS_MODEL = "deepseek-v4-pro";

export const _SUM_DEF =
  "你是一名专业的内容分析师。请阅读以下文本，生成一份简洁的背景摘要，内容包括：主要话题、关键术语、角色关系以及整体基调。用中文输出，不超过 150 字。";

export const _TRANS_DEF = `你是一名专业的影视字幕翻译专家，负责将英文字幕翻译成简体中文。

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
8. 句末不加句号（。）
9. 必须保留问号（？）和叹号（！）
10. 句中停顿用空格代替逗号（，）

## JSON 格式示例：
{"translations": [{"id": 0, "text": "你好"}, {"id": 1, "text": "世界"}]}`;

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function useSettings() {
  const [apiKey, setApiKey] = useState(() => load("apiKey", ""));
  const [url, setUrl] = useState(() => load("url", DEF_URL));
  const [sumModel, setSumModel] = useState(() => load("sumModel", DEF_SUM_MODEL));
  const [transModel, setTransModel] = useState(() => load("transModel", DEF_TRANS_MODEL));
  const [sumPrompt, setSumPrompt] = useState(() => load("sumPrompt", _SUM_DEF));
  const [transPrompt, setTransPrompt] = useState(() => load("transPrompt", _TRANS_DEF));
  const [savePath, setSavePath] = useState(() => load("savePath", ""));
  const [glossary, setGlossary] = useState(() => load("glossary", ""));
  const [concurrency, setConcurrency] = useState(() => load("concurrency", 8));

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounced localStorage sync: 500ms after last change
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      localStorage.setItem("apiKey", JSON.stringify(apiKey));
      localStorage.setItem("url", JSON.stringify(url));
      localStorage.setItem("sumModel", JSON.stringify(sumModel));
      localStorage.setItem("transModel", JSON.stringify(transModel));
      localStorage.setItem("sumPrompt", JSON.stringify(sumPrompt));
      localStorage.setItem("transPrompt", JSON.stringify(transPrompt));
      localStorage.setItem("savePath", JSON.stringify(savePath));
      localStorage.setItem("glossary", JSON.stringify(glossary));
      localStorage.setItem("concurrency", JSON.stringify(concurrency));
    }, 500);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [apiKey, url, sumModel, transModel, sumPrompt, transPrompt, savePath, glossary, concurrency]);

  const resetToDefaults = () => {
    setApiKey("");
    setUrl(DEF_URL);
    setSumModel(DEF_SUM_MODEL);
    setTransModel(DEF_TRANS_MODEL);
    setSumPrompt(_SUM_DEF);
    setTransPrompt(_TRANS_DEF);
    setSavePath("");
    setGlossary("");
    setConcurrency(8);
  };

  return {
    apiKey, setApiKey,
    url, setUrl,
    sumModel, setSumModel,
    transModel, setTransModel,
    sumPrompt, setSumPrompt,
    transPrompt, setTransPrompt,
    savePath, setSavePath,
    glossary, setGlossary,
    concurrency, setConcurrency,
    resetToDefaults,
  };
}
