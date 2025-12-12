# AI Subtitle Translator

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/LLM-DeepSeek%2FOpenAI-412991?logo=openai&logoColor=white)
![SpaCy](https://img.shields.io/badge/NLP-SpaCy-09A3D5?logo=spacy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **基于 LLM 的高性能异步字幕翻译工具**
>
> 结合 `asyncio` 高并发、全局上下文感知与 NLP 智能断句，为长视频提供精准、连贯的 SRT 字幕翻译体验。

---

## ✨ 核心特性

- **⚡ 异步高并发 (Async I/O)**
  基于 `asyncio` 和 `tqdm` 实现多协程并发，大幅缩短长视频翻译耗时，告别漫长等待。

- **🧠 全局上下文感知 (Context-Aware)**
  - **剧情摘要**：预先生成全文背景摘要，让 AI "读懂" 故事大纲。
  - **滑动窗口**：翻译时自动携带前后文，确保语境连贯，拒绝"机翻味"。

- **🔗 智能断句合并 (Smart Merge)**
  内置 **SpaCy** NLP 引擎，自动识别并合并被时间轴切碎的短句，修复语义断层。

- **📚 动态术语表 (Dynamic Glossary)**
  支持自定义 `glossary.txt`，采用 RAG 风格的动态注入机制，确保专有名词翻译统一。

- **🛠️ 自动清洗 (Auto-Clean)**
  智能移除多余编号、修正不规范标点，输出即用的高质量字幕。
