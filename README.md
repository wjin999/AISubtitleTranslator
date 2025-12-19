# SRT Translator 🎬

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

An async LLM-powered subtitle translator with intelligent merging capabilities. Translate your SRT subtitle files to Chinese with context-aware translations and consistent terminology.

### ✨ Features

- **🧠 Intelligent Merging**: Uses spaCy NLP to smartly merge fragmented subtitles while preserving sentence boundaries
- **📖 Context-Aware Translation**: Generates a content summary and provides surrounding context to the LLM for more accurate translations
- **📚 Glossary Support**: Maintain consistent translations with custom terminology glossaries
- **⚡ Async Processing**: High-performance concurrent API calls with configurable rate limiting
- **🎯 Dynamic Term Matching**: Only injects relevant glossary terms into each translation chunk

### 📦 Installation

#### From Source

```bash
git clone https://github.com/yourusername/srt-translator.git
cd srt-translator
pip install -e .
```

#### With Smart Merging (spaCy)

```bash
pip install -e ".[merge]"
python -m spacy download en_core_web_sm
```

#### Development Installation

```bash
pip install -e ".[all]"
```

### 🚀 Quick Start

#### 1. Set up your API key

```bash
# Option 1: Environment variable
export DEEPSEEK_API_KEY="your_api_key_here"

# Option 2: Create .env file
cp .env.example .env
# Edit .env with your API key
```

#### 2. Translate a subtitle file

```bash
# Basic usage
srt-translator video.srt

# Specify output file
srt-translator video.srt -o translated.srt

# With custom glossary
srt-translator video.srt --glossary my_terms.txt

# Disable smart merging
srt-translator video.srt --no-merge
```

### 📖 CLI Options

```
usage: srt-translator [-h] [--glossary GLOSSARY_PATH] [--no-merge]
                      [--max-chars MAX_CHARS] [--merge-gap MERGE_GAP]
                      [--api-key API_KEY] [--base-url BASE_URL]
                      [--model MODEL] [--summary-model SUMMARY_MODEL]
                      [--concurrency CONCURRENCY] [--chunk-size CHUNK_SIZE]
                      [--verbose]
                      input_path [output_path]

positional arguments:
  input_path            Path to input SRT file
  output_path           Path to output SRT file (default: translated_<input>)

options:
  -h, --help            Show help message
  --glossary, -g        Path to glossary file (format: Term=Translation)
  --no-merge            Disable smart merging (translate line-by-line)
  --max-chars           Maximum characters per merged entry (default: 300)
  --merge-gap           Maximum time gap (seconds) for merging (default: 1.5)
  --api-key             API key (default: DEEPSEEK_API_KEY env var)
  --base-url            API base URL (default: https://api.deepseek.com)
  --model               Translation model (default: deepseek-chat)
  --summary-model       Summary model (default: deepseek-reasoner)
  --concurrency         Max concurrent API requests (default: 8)
  --chunk-size          Lines per translation chunk (default: 10)
  --verbose, -v         Enable verbose logging
```

### 📚 Glossary Format

Create a text file with one term per line in `Term = Translation` format:

```
# Character Names
John = 约翰
Dr. Smith = 史密斯博士

# Technical Terms
artificial intelligence = 人工智能
machine learning = 机器学习
```

The translator automatically detects `glossary.txt` in the current directory.

### 🔧 Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `concurrency` | 8 | Max concurrent API requests |
| `chunk_size` | 10 | Lines per translation batch |
| `max_chars` | 300 | Max characters per merged entry |
| `merge_gap` | 1.5s | Max time gap for merging |

---

<a name="中文"></a>
## 中文

基于 LLM 的异步字幕翻译工具，支持智能合并功能。可将 SRT 字幕文件翻译成中文，具备上下文感知翻译和术语一致性保证。

### ✨ 功能特性

- **🧠 智能合并**：使用 spaCy NLP 智能合并碎片化字幕，同时保留句子边界
- **📖 上下文感知翻译**：生成内容摘要并为 LLM 提供上下文，实现更准确的翻译
- **📚 术语表支持**：通过自定义术语表保持翻译一致性
- **⚡ 异步处理**：高性能并发 API 调用，支持可配置的速率限制
- **🎯 动态术语匹配**：仅将相关术语注入每个翻译块

### 📦 安装

#### 从源码安装

```bash
git clone https://github.com/yourusername/srt-translator.git
cd srt-translator
pip install -e .
```

#### 启用智能合并（需要 spaCy）

```bash
pip install -e ".[merge]"
python -m spacy download en_core_web_sm
```

#### 开发环境安装

```bash
pip install -e ".[all]"
```

### 🚀 快速开始

#### 1. 配置 API 密钥

```bash
# 方式一：环境变量
export DEEPSEEK_API_KEY="your_api_key_here"

# 方式二：创建 .env 文件
cp .env.example .env
# 编辑 .env 填入你的 API 密钥
```

#### 2. 翻译字幕文件

```bash
# 基本用法
srt-translator video.srt

# 指定输出文件
srt-translator video.srt -o translated.srt

# 使用自定义术语表
srt-translator video.srt --glossary my_terms.txt

# 禁用智能合并
srt-translator video.srt --no-merge
```

### 📖 命令行参数

```
用法: srt-translator [-h] [--glossary 术语表路径] [--no-merge]
                     [--max-chars 最大字符数] [--merge-gap 合并间隔]
                     [--api-key API密钥] [--base-url API地址]
                     [--model 模型名] [--summary-model 摘要模型]
                     [--concurrency 并发数] [--chunk-size 块大小]
                     [--verbose]
                     输入文件 [输出文件]

位置参数:
  input_path            输入 SRT 文件路径
  output_path           输出 SRT 文件路径（默认: translated_<输入文件名>）

可选参数:
  -h, --help            显示帮助信息
  --glossary, -g        术语表文件路径（格式: 术语=翻译）
  --no-merge            禁用智能合并（逐行翻译）
  --max-chars           合并后每条字幕最大字符数（默认: 300）
  --merge-gap           合并的最大时间间隔（秒）（默认: 1.5）
  --api-key             API 密钥（默认读取 DEEPSEEK_API_KEY 环境变量）
  --base-url            API 基础地址（默认: https://api.deepseek.com）
  --model               翻译模型（默认: deepseek-chat）
  --summary-model       摘要模型（默认: deepseek-reasoner）
  --concurrency         最大并发请求数（默认: 8）
  --chunk-size          每次翻译的行数（默认: 10）
  --verbose, -v         启用详细日志
```

### 📚 术语表格式

创建一个文本文件，每行一个术语，格式为 `英文术语 = 中文翻译`：

```
# 人物名称
John = 约翰
Dr. Smith = 史密斯博士

# 专业术语
artificial intelligence = 人工智能
machine learning = 机器学习

# 网络用语
awesome = 绝绝子
cool = 牛逼
```

程序会自动检测当前目录下的 `glossary.txt` 文件。

### 🔧 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `concurrency` | 8 | 最大并发 API 请求数 |
| `chunk_size` | 10 | 每批翻译的字幕行数 |
| `max_chars` | 300 | 合并后每条字幕最大字符数 |
| `merge_gap` | 1.5秒 | 合并的最大时间间隔 |

---

## 📁 Project Structure / 项目结构

```
srt-translator/
├── src/
│   └── srt_translator/
│       ├── __init__.py      # Package exports / 包导出
│       ├── __main__.py      # python -m entry / 模块入口
│       ├── cli.py           # CLI interface / 命令行接口
│       ├── config.py        # Configuration / 配置管理
│       ├── models.py        # Data models / 数据模型
│       ├── parser.py        # SRT parsing / SRT 解析
│       ├── merger.py        # Smart merging / 智能合并
│       ├── glossary.py      # Glossary loading / 术语表加载
│       ├── translator.py    # Translation logic / 翻译逻辑
│       ├── llm_client.py    # API client / API 客户端
│       └── text_utils.py    # Text utilities / 文本工具
├── tests/                   # Unit tests / 单元测试
├── pyproject.toml           # Package config / 包配置
├── README.md
├── LICENSE
├── .env.example
├── .gitignore
└── glossary.example.txt
```

## 🧪 Development / 开发

### Run Tests / 运行测试

```bash
pytest
```

### Code Formatting / 代码格式化

```bash
black src/ tests/
isort src/ tests/
```

### Type Checking / 类型检查

```bash
mypy src/
```

## 🤝 Contributing / 贡献

Contributions are welcome! / 欢迎贡献代码！

1. Fork the repository / Fork 本仓库
2. Create your feature branch / 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. Commit your changes / 提交更改 (`git commit -m 'Add some amazing feature'`)
4. Push to the branch / 推送分支 (`git push origin feature/amazing-feature`)
5. Open a Pull Request / 提交 PR

## 📄 License / 许可证

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments / 致谢

- [DeepSeek](https://www.deepseek.com/) - LLM API
- [spaCy](https://spacy.io/) - NLP capabilities / NLP 支持
- [OpenAI Python Client](https://github.com/openai/openai-python) - API interaction / API 交互
