# -*- coding: utf-8 -*-
import os

# README 的内容
readme_content = r"""# Async LLM Subtitle Translator Pro

这是一个基于大语言模型（LLM）的高性能字幕翻译工具。它不仅仅是简单地逐行翻译，还结合了**上下文感知**、**术语表管理**以及基于 **spaCy** 的智能断句合并功能，旨在提供更加流畅、准确的字幕翻译体验。

默认配置针对 **DeepSeek** API 进行了优化，但也支持任何兼容 OpenAI 格式的 API。

## ✨ 主要功能

* **🚀 异步高并发**: 使用 Python `asyncio` 和 `Semaphore` 实现多线程并发翻译，大幅提高处理速度。

* **🧠 上下文感知 (Context-Aware)**:

  * **全局摘要**: 在翻译前，首先使用推理模型（如 `deepseek-reasoner`）阅读全文并生成背景摘要，帮助翻译模型理解剧情语气。

  * **滑动窗口**: 翻译每一块字幕时，会自动带入前文和后文作为参考，避免“语境缺失”。

* **🔗 智能合并 (Smart Merging)**: 利用 `spaCy` NLP 模型分析句子结构，将短小、破碎的时间轴合并为完整的句子后再翻译，减少翻译碎片化，提升阅读体验。

* **📚 术语表支持 (Glossary)**: 支持加载自定义术语表，确保人名、地名、专有名词翻译的一致性。支持自动检测目录下的 `glossary.txt`。

* **🛠️ 鲁棒性设计**: 内置 API 速率限制处理和自动重试机制。

## 📦 安装与依赖

确保你的 Python 版本 >= 3.8。

1. **克隆或下载代码**

2. **安装依赖库**

   ```bash
   pip install openai tqdm python-dotenv spacy
   ```

3. **下载 spaCy 语言模型** (用于智能合并功能)

   ```bash
   python -m spacy download en_core_web_sm
   ```

## ⚙️ 配置与使用

### 1. 设置 API Key

在脚本同级目录下创建一个 `.env` 文件，填入你的 API Key：

```env
DEEPSEEK_API_KEY=sk-your-api-key-here
```

或者在运行时通过命令行参数传入。

### 2. 准备术语表 (可选)

你可以创建一个 `glossary.txt` 文件，格式为 `原文 = 译文`：

```text
John Doe = 强子
New York = 纽约
Cyberpunk = 赛博朋克
```

*脚本会自动检测当前目录下的 `glossary.txt`，或者通过 `--glossary` 指定路径。*

### 3. 运行命令

#### 基本用法

```bash
python translate_srt.py input.srt
```

这将生成 `translated_input.srt`。

#### 常用参数示例

```bash
# 指定输出路径和术语表
python translate_srt.py input.srt output.srt --glossary my_terms.txt

# 禁用智能合并 (逐行翻译模式)
python translate_srt.py input.srt --no-merge

# 使用其他模型 (例如 GPT-4o) 并调整并发数
python translate_srt.py input.srt --model gpt-4o --concurrency 5 --base-url [https://api.openai.com/v1](https://api.openai.com/v1)
```

## 📋 参数说明

| **参数** | **简写** | **说明** | **默认值** |
| :--- | :--- | :--- | :--- |
| `input_path` | - | **(必填)** 输入的 SRT 字幕文件路径 | - |
| `output_path` | - | 输出文件路径 | `translated_[文件名]` |
| `--glossary` | - | 术语表文件路径 | 自动检测 `glossary.txt` |
| `--no-merge` | - | 禁用 spaCy 智能合并功能 | False (默认启用) |
| `--api-key` | - | API 密钥 (优先于环境变量) | - |
| `--base-url` | - | API 基础地址 | `https://api.deepseek.com` |
| `--model` | - | 用于翻译的聊天模型 | `deepseek-chat` |
| `--summary-model` | - | 用于生成摘要的推理/更强模型 | `deepseek-reasoner` |
| `--concurrency` | - | 并发请求数量 | 8 |
| `--chunk-size` | - | 每次请求翻译的字幕行数 | 10 |
| `--max-chars` | - | 合并时允许的最大字符数 | 300 |
| `--merge-gap` | - | 合并时允许的最大时间间隔(秒) | 1.5 |

## 💡 工作原理

1. **解析**: 读取 SRT 文件。

2. **加载术语**: 读取术语表，后续动态注入 Prompt。

3. **智能合并**: (除非禁用) 使用 spaCy 分析文本，将时间间隔短且属于同一句子的字幕行合并。

4. **摘要生成**: 抽取全文内容，调用 `deepseek-reasoner` 生成剧情/背景摘要。

5. **切分与翻译**:

   * 将字幕按 `chunk-size` 分组。

   * **动态构建 Prompt**: 包含全局摘要 + 上下文(前2行/后2行) + **当前块中匹配到的术语**。

   * 异步发送请求。

6. **后处理**: 解析 JSON 返回，清洗标点符号。

7. **保存**: 输出新的 SRT 文件。

## ⚠️ 注意事项

* **API 成本**: 该工具会消耗 Token，特别是开启智能合并和摘要功能时。请关注你的 API 使用量。

* **JSON 模式**: 脚本强制要求模型返回 JSON 格式。如果模型不支持 JSON Mode (如旧版模型)，可能会导致解析失败。建议使用较新的模型版本。

## License

MIT License
"""

def create_readme():
    file_path = "README.md"
    try:
        # 使用 utf-8 编码写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(readme_content.strip())
        print(f"✅ 成功生成文件: {os.path.abspath(file_path)}")
        print("现在你可以将此文件上传到 GitHub，它会自动显示格式。")
    except Exception as e:
        print(f"❌ 生成失败: {e}")

if __name__ == "__main__":
    create_readme()
