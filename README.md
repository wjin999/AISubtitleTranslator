<div align="center">

# SRT Translator

**基于 LLM 的智能字幕翻译工具**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](./README_EN.md) | 简体中文

</div>

---

## 简介

SRT Translator 是一个基于大语言模型的异步字幕翻译工具。它能够将英文 SRT 字幕文件翻译成中文，支持智能合并碎片化字幕、上下文感知翻译、自定义术语表等功能。

## 特性

- 🧠 **智能合并** - 使用 spaCy NLP 分析句子边界，智能合并碎片化字幕
- 📖 **上下文感知** - 自动生成内容摘要，为翻译提供全局上下文
- 📚 **术语表支持** - 自定义术语表确保专有名词翻译一致
- ⚡ **异步并发** - 高性能异步处理，支持并发数配置
- 🎯 **动态匹配** - 仅注入当前文本块相关的术语

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/srt-translator.git
cd srt-translator

# 基础安装
pip install -e .

# 启用智能合并（可选）
pip install -e ".[merge]"
python -m spacy download en_core_web_sm
```

## 快速开始

### 配置 API 密钥

```bash
export DEEPSEEK_API_KEY="your_api_key"
```

或创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_api_key
```

### 基本用法

```bash
# 翻译字幕
srt-translator video.srt

# 指定输出
srt-translator video.srt -o output.srt

# 使用术语表
srt-translator video.srt -g glossary.txt

# 禁用合并
srt-translator video.srt --no-merge
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_path` | 输入 SRT 文件 | 必填 |
| `-o, --output` | 输出文件路径 | `translated_*.srt` |
| `-g, --glossary` | 术语表文件 | `glossary.txt` |
| `--no-merge` | 禁用智能合并 | `False` |
| `--max-chars` | 合并最大字符数 | `300` |
| `--merge-gap` | 合并最大时间间隔(秒) | `1.5` |
| `--model` | 翻译模型 | `deepseek-chat` |
| `--concurrency` | 并发数 | `8` |
| `--chunk-size` | 每批行数 | `10` |
| `-v, --verbose` | 详细日志 | `False` |

## 术语表格式

```
# 人名
John = 约翰
Dr. Smith = 史密斯博士

# 术语
machine learning = 机器学习
```

## 项目结构

```
srt-translator/
├── src/srt_translator/
│   ├── cli.py           # 命令行入口
│   ├── config.py        # 配置管理
│   ├── models.py        # 数据模型
│   ├── parser.py        # SRT 解析
│   ├── merger.py        # 智能合并
│   ├── glossary.py      # 术语表
│   ├── translator.py    # 翻译核心
│   └── llm_client.py    # API 客户端
├── tests/               # 测试
├── pyproject.toml
└── README.md
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[all]"

# 运行测试
pytest

# 代码格式化
black src/ tests/
```

## 许可证

[MIT License](LICENSE)

## 致谢

- [DeepSeek](https://www.deepseek.com/) - LLM API
- [spaCy](https://spacy.io/) - NLP 引擎
