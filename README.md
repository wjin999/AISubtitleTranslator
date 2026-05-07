# AISubtitleTranslator

基于大语言模型的异步视频字幕翻译工具，支持智能合并、术语表、进度保存/断点续翻。

## 功能特性

- 🧠 基于 LLM 的上下文感知翻译
- 🔗 智能句子合并（基于 spaCy NLP）
- 📚 术语表支持，确保术语一致性
- ⚡ 异步并发处理，高性能
- 💾 进度保存与断点续翻
- 🖥️ CLI + Web API + Tauri 桌面应用三合一

## 快速开始

### 安装

```bash
pip install srt-translator

# 可选依赖
pip install "srt-translator[merge]"   # 智能合并（需 spaCy）
pip install "srt-translator[server]"  # Web API 服务
pip install "srt-translator[all]"     # 全部功能
```

### CLI 使用

```bash
# 设置 API Key
export DEEPSEEK_API_KEY=your_key_here

# 基本翻译
srt-translator input.srt

# 指定输出
srt-translator input.srt -o output.srt

# 使用术语表
srt-translator input.srt -g glossary.txt

# 禁用自动合并
srt-translator input.srt --no-merge
```

### API 服务

```bash
pip install "srt-translator[server]"
python api_server.py
```

启动后访问 `http://127.0.0.1:8000/docs` 查看 API 文档。

### 桌面应用

见 [ui/README.md](ui/README.md)

## 配置

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `DEEPSEEK_API_KEY` | API 密钥 | - |
| `DEEPSEEK_MODEL` | 翻译模型 | deepseek-chat |
| `DEEPSEEK_SUMMARY_MODEL` | 摘要模型 | deepseek-chat |

## 术语表格式

每行一个术语，支持以下分隔符：

```
原文 = 译文
原文 -> 译文
原文：译文
```

## 许可证

MIT
