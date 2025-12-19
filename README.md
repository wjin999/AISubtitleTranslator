# SRT Translator 🎬

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An async LLM-powered subtitle translator with intelligent merging capabilities. Translate your SRT subtitle files to Chinese with context-aware translations and consistent terminology.

## ✨ Features

- **🧠 Intelligent Merging**: Uses spaCy NLP to smartly merge fragmented subtitles while preserving sentence boundaries
- **📖 Context-Aware Translation**: Generates a content summary and provides surrounding context to the LLM for more accurate translations
- **📚 Glossary Support**: Maintain consistent translations with custom terminology glossaries
- **⚡ Async Processing**: High-performance concurrent API calls with configurable rate limiting
- **🎯 Dynamic Term Matching**: Only injects relevant glossary terms into each translation chunk

## 📦 Installation

### From PyPI (recommended)

```bash
pip install srt-translator
```

### From Source

```bash
git clone https://github.com/yourusername/srt-translator.git
cd srt-translator
pip install -e .
```

### With Smart Merging (spaCy)

```bash
pip install srt-translator[merge]
python -m spacy download en_core_web_sm
```

### Development Installation

```bash
pip install -e ".[all]"
```

## 🚀 Quick Start

### 1. Set up your API key

```bash
# Option 1: Environment variable
export DEEPSEEK_API_KEY="your_api_key_here"

# Option 2: Create .env file
cp .env.example .env
# Edit .env with your API key
```

### 2. Translate a subtitle file

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

## 📖 Usage

### Command Line Interface

```
usage: srt-translator [-h] [--glossary GLOSSARY_PATH] [--no-merge]
                      [--max-chars MAX_CHARS] [--merge-gap MERGE_GAP]
                      [--api-key API_KEY] [--base-url BASE_URL]
                      [--model MODEL] [--summary-model SUMMARY_MODEL]
                      [--concurrency CONCURRENCY] [--chunk-size CHUNK_SIZE]
                      [--verbose]
                      input_path [output_path]

Async LLM Subtitle Translator with Smart Merging

positional arguments:
  input_path            Path to input SRT file
  output_path           Path to output SRT file (default: translated_<input>)

options:
  -h, --help            show this help message and exit
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

### Python API

```python
import asyncio
from srt_translator import (
    parse_srt, 
    save_srt, 
    merge_entries, 
    init_spacy_model,
    load_glossary
)

# Parse SRT file
with open("video.srt", encoding="utf-8") as f:
    entries = parse_srt(f.read())

# Optional: Initialize spaCy and merge entries
init_spacy_model()
merged = merge_entries(entries, max_chars=300, time_gap_threshold=1.5)

# Load glossary
glossary = load_glossary("glossary.txt")

# ... perform translation ...

# Save result
save_srt(translated_entries, "output.srt")
```

## 📚 Glossary Format

Create a text file with one term per line in `Term = Translation` format:

```
# Character Names
John = 约翰
Dr. Smith = 史密斯博士

# Technical Terms
artificial intelligence = 人工智能
machine learning = 机器学习

# Common Phrases
thank you = 谢谢
```

The translator automatically detects if a `glossary.txt` file exists in the current directory.

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPSEEK_API_KEY` | API key for DeepSeek | (required) |

### Default Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `concurrency` | 8 | Max concurrent API requests |
| `chunk_size` | 10 | Lines per translation batch |
| `max_chars` | 300 | Max characters per merged entry |
| `merge_gap` | 1.5s | Max time gap for merging |

## 📁 Project Structure

```
srt-translator/
├── src/
│   └── srt_translator/
│       ├── __init__.py      # Package exports
│       ├── __main__.py      # python -m entry point
│       ├── cli.py           # Command-line interface
│       ├── config.py        # Configuration management
│       ├── models.py        # Data models (SrtEntry)
│       ├── parser.py        # SRT parsing/saving
│       ├── merger.py        # spaCy-based merging
│       ├── glossary.py      # Glossary loading
│       ├── translator.py    # LLM translation logic
│       ├── llm_client.py    # API client utilities
│       └── text_utils.py    # Text processing
├── tests/
│   ├── test_models.py
│   ├── test_parser.py
│   └── test_glossary.py
├── pyproject.toml
├── README.md
├── LICENSE
├── .env.example
├── .gitignore
└── glossary.example.txt
```

## 🧪 Development

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

### Type Checking

```bash
mypy src/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DeepSeek](https://www.deepseek.com/) for the LLM API
- [spaCy](https://spacy.io/) for NLP capabilities
- [OpenAI Python Client](https://github.com/openai/openai-python) for API interaction
