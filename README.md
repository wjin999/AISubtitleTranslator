LLM Subtitle Translator Pro
🚀 An intelligent, async, context-aware subtitle translation tool powered by LLMs (DeepSeek/OpenAI) and NLP.

LLM Subtitle Translator Pro 是一个高性能的 SRT 字幕翻译 CLI 工具。与传统的逐行翻译不同，它利用 spaCy 进行智能句法分析以合并断句，利用 AsyncIO 实现高并发请求，并引入了 双向上下文感知 和 动态术语表 机制，旨在生成流畅、地道且术语一致的专业级字幕。

✨ Features
🚀 Async High-Performance: 基于 Python asyncio 和 tqdm 实现多协程并发，翻译速度比传统串行脚本快 5-10 倍。

🧠 Smart NLP Merging: 集成 spaCy 语言模型，结合时间阈值（1.5s）和句法分析，自动修复时间轴切分导致的“断句”问题（如 that have + turned into）。

📖 Context-Aware Translation: 翻译时自动注入全局剧情摘要、上文和下文，有效解决代词指代（It/They）歧义和语气连贯性问题。

📚 Dynamic Glossary Support: 支持自动加载或手动指定术语表（Term=Translation），仅将当前片段涉及的术语注入 Prompt，节省 Token 并保证一致性。

🛡️ Robust JSON Output: 强制 LLM 输出结构化 JSON 数据，彻底解决漏行、错行和格式混乱问题。

🛠 Tech Stack
Core: Python 3.10+

LLM Integration: OpenAI SDK (Compatible with DeepSeek, OpenAI, etc.)

NLP Engine: spaCy (en_core_web_sm)

Concurrency: asyncio, tqdm

Config: python-dotenv

⚡ Quick Start
Prerequisites
Python 3.10 or higher

An API Key (DeepSeek recommended for cost/performance)

Installation
Clone the repository

Bash

git clone https://github.com/your-username/llm-subtitle-translator.git
cd llm-subtitle-translator
Install dependencies

Bash

pip install openai tqdm spacy python-dotenv
Download the NLP model

Bash

python -m spacy download en_core_web_sm
Configure API Key Create a .env file in the root directory:

代码段

DEEPSEEK_API_KEY=sk-your_api_key_here
💻 Usage Examples
1. Basic Usage (Auto-Merge & Auto-Glossary)
The script automatically detects glossary.txt in the current directory and enables smart merging by default.

Bash

# Simply run with the subtitle file
python translate.py movie.srt
2. Custom Terminology (Manual Glossary)
Specify a custom glossary file (Format: English Term = Chinese Translation).

Bash

python translate.py video.srt --glossary valorant_terms.txt
3. Lyric/Music Mode (No Merge)
Disable smart merging for content that requires strict line-by-line translation, such as lyrics or poems.

Bash

python translate.py song.srt --no-merge
4. Power User Configuration
Customize concurrency, model, and chunk size for maximum performance.

Bash

python translate.py video.srt \
  --model deepseek-chat \
  --concurrency 10 \
  --chunk-size 15 \
  --output_path translated_final.srt
📂 Project Structure
Plaintext

.
├── translate.py           # Main script (The code provided)
├── glossary.txt           # (Optional) Terminology mapping file
├── .env                   # API Key configuration (Do not commit this!)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
📄 License
Distributed under the MIT0 License. See LICENSE for more information.
