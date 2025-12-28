# SRT Translator

基于 LLM 的英文字幕翻译工具，支持智能合并和断点续传。

## 安装

```bash
git clone https://github.com/wjin999/srt-translator.git
cd srt-translator
pip install -e ".[merge]"
python -m spacy download en_core_web_sm
```

## 使用

```bash
# 设置 API Key
export DEEPSEEK_API_KEY="sk-xxx"

# 翻译
srt-translator video.srt

# 指定输出
srt-translator video.srt -o output.srt

# 使用术语表
srt-translator video.srt -g glossary.txt

# 断点续传
srt-translator video.srt --resume
```

## 术语表格式

```
# 人名
John = 约翰

# 术语
machine learning = 机器学习
```

## 功能

- **智能合并** - 用 spaCy 分析句子边界，合并碎片字幕
- **上下文翻译** - 提供前后文和全局摘要，提高翻译质量
- **术语表** - 确保专有名词翻译一致
- **断点续传** - 中断后可继续翻译

## 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-o` | 输出文件 | translated_*.srt |
| `-g` | 术语表文件 | glossary.txt |
| `--no-merge` | 禁用合并 | - |
| `--resume` | 断点续传 | - |
| `--model` | 翻译模型 | deepseek-chat |
| `--concurrency` | 并发数 | 8 |
| `-v` | 详细日志 | - |

## License

MIT
