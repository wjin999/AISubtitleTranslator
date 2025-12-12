# AI Subtitle Translator

> 🚀 **一款基于大语言模型（LLM）与 NLP 技术的智能字幕翻译工具。**

**AI Subtitle Translator** 是一个高性能的 SRT 字幕翻译 CLI 工具。它结合了 **Python AsyncIO** 的高并发能力与 **spaCy** 的自然语言处理能力，旨在解决传统机器翻译中的“断句错误”和“代词歧义”问题。通过双向上下文感知和动态术语表注入，它能生成流畅、地道且术语一致的专业级中文字幕。

## ✨ Features (核心特性)

* 🚀 **Async High-Performance**: 基于 Python `asyncio` 和 `tqdm` 实现多协程并发请求，翻译速度相比传统串行脚本提升 **5-10 倍**。
* 🧠 **Smart NLP Merging**: 集成 `spaCy` 语言模型，结合时间阈值（1.5s）和句法分析，自动合并被时间轴切断的长难句（如 `that have` + `turned into`），确保翻译语义完整。
* 📖 **Context-Aware Translation**: 翻译时自动注入**全局剧情摘要**、**上文**和**下文**，有效解决多义词、代词指代（It/They）及语气连贯性问题。
* 📚 **Dynamic Glossary Support**: 支持自动检测或手动指定术语表（`Term=Translation`），仅将当前片段涉及的术语注入 Prompt，节省 Token 并保证专业术语一致性。
* 🛡️ **Robust JSON Output**: 强制 LLM 输出结构化 JSON 数据，彻底解决漏行、错行和格式混乱问题。
