<div align="center">

# SRT Translator

**LLM-Powered Intelligent Subtitle Translator**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

English | [简体中文](./README.md)

</div>

---

## Introduction

SRT Translator is an async subtitle translation tool powered by Large Language Models. It translates English SRT subtitle files to Chinese with intelligent subtitle merging, context-aware translation, and custom glossary support.

## Features

- 🧠 **Smart Merging** - Uses spaCy NLP to analyze sentence boundaries and merge fragmented subtitles
- 📖 **Context-Aware** - Auto-generates content summary to provide global context for translation
- 📚 **Glossary Support** - Custom glossary ensures consistent translation of proper nouns
- ⚡ **Async Processing** - High-performance async processing with configurable concurrency
- 🎯 **Dynamic Matching** - Only injects relevant glossary terms for each text chunk

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/srt-translator.git
cd srt-translator

# Basic installation
pip install -e .

# Enable smart merging (optional)
pip install -e ".[merge]"
python -m spacy download en_core_web_sm
```

## Quick Start

### Configure API Key

```bash
export DEEPSEEK_API_KEY="your_api_key"
```

Or create a `.env` file:

```
DEEPSEEK_API_KEY=your_api_key
```

### Basic Usage

```bash
# Translate subtitles
srt-translator video.srt

# Specify output
srt-translator video.srt -o output.srt

# Use glossary
srt-translator video.srt -g glossary.txt

# Disable merging
srt-translator video.srt --no-merge
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_path` | Input SRT file | Required |
| `-o, --output` | Output file path | `translated_*.srt` |
| `-g, --glossary` | Glossary file | `glossary.txt` |
| `--no-merge` | Disable smart merging | `False` |
| `--max-chars` | Max chars for merging | `300` |
| `--merge-gap` | Max time gap for merging (sec) | `1.5` |
| `--model` | Translation model | `deepseek-chat` |
| `--concurrency` | Concurrency level | `8` |
| `--chunk-size` | Lines per batch | `10` |
| `-v, --verbose` | Verbose logging | `False` |

## Glossary Format

```
# Names
John = 约翰
Dr. Smith = 史密斯博士

# Terms
machine learning = 机器学习
```

## Project Structure

```
srt-translator/
├── src/srt_translator/
│   ├── cli.py           # CLI entry
│   ├── config.py        # Configuration
│   ├── models.py        # Data models
│   ├── parser.py        # SRT parsing
│   ├── merger.py        # Smart merging
│   ├── glossary.py      # Glossary
│   ├── translator.py    # Translation core
│   └── llm_client.py    # API client
├── tests/               # Tests
├── pyproject.toml
└── README.md
```

## Development

```bash
# Install dev dependencies
pip install -e ".[all]"

# Run tests
pytest

# Format code
black src/ tests/
```

## License

[MIT License](LICENSE)

## Acknowledgments

- [DeepSeek](https://www.deepseek.com/) - LLM API
- [spaCy](https://spacy.io/) - NLP Engine
