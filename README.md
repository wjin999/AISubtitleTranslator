AI Subtitle Translator
这是一个基于 Python 的高性能异步字幕翻译工具。它利用大语言模型（LLM，如 DeepSeek、OpenAI）的强大能力，结合上下文理解、术语表支持和智能断句合并功能，提供高质量的 SRT 字幕翻译体验。

核心功能
⚡ 异步高并发：使用 asyncio 和 tqdm 实现多线程并发翻译，大幅提升长视频字幕的翻译速度。

🧠 全局上下文感知：

剧情摘要：在翻译前先生成全文背景摘要，让 AI 了解故事大纲。

滑动窗口：翻译每一块字幕时，会自动带上前文和后文，确保语境连贯。

🔗 智能断句合并 (SpaCy)：利用 spaCy NLP 模型，自动检测并合并被时间轴切碎的短句，解决"一句话被切成两行导致翻译破碎"的问题。

📚 动态术语表：支持加载自定义术语表（Glossary），并在翻译时动态注入 Prompt，确保人名、地名、专有名词翻译一致。

🛠️ 自动标点处理：自动清洗翻译结果中多余的编号、特殊符号和不规范标点。

安装指南
环境要求 Python 3.8+

安装依赖库 运行以下命令安装所需的 Python 包： pip install openai tqdm python-dotenv spacy

下载 spaCy 语言模型 为了使用智能断句合并功能，需要下载英文语言模型： python -m spacy download en_core_web_sm

快速开始
配置 API Key 推荐在项目根目录下创建一个 .env 文件，填入你的 API 密钥。默认配置为 DeepSeek API，也可以通过参数修改为其他兼容 OpenAI 格式的 API。

.env 文件内容示例： DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

准备术语表 (可选) 在同级目录下创建一个 glossary.txt 文件，格式为 "原文 = 译文"：

John Doe = 约翰·多伊 Cyberpunk = 赛博朋克 Warp Drive = 曲速引擎

运行翻译 最简单的运行方式： python translate.py input.srt

程序会自动生成 translated_input.srt。

详细使用说明
[ 命令行参数 ]

input_path (必填) 输入的 SRT 字幕文件路径。

output_path (可选) 输出文件路径。默认值：translated_[文件名]

--glossary 指定术语表文件路径。默认值：自动检测 glossary.txt

--no-merge 禁用 spaCy 智能合并，按原行翻译。默认：开启

--api-key 手动指定 API Key (覆盖 .env)。

--base-url API 基础地址。默认值：https://api.deepseek.com

--model 用于翻译的主模型名称。默认值：deepseek-chat

--summary-model 用于生成摘要的模型名称。默认值：deepseek-reasoner

--concurrency 并发请求数量。默认值：8

--chunk-size 每次请求包含的字幕行数。默认值：10

[ 典型使用示例 ]

指定自定义术语表： python translate.py movie.srt --glossary terms.txt

使用其他模型 (如 GPT-4o)： python translate.py movie.srt --base-url https://api.openai.com/v1 --model gpt-4o --api-key sk-xxxx

仅逐行翻译 (不合并句子)： 如果你的字幕时间轴非常精准，不需要合并碎句，可以使用此选项： python translate.py interview.srt --no-merge

调整并发以避免限流： 如果遇到 API Rate Limit 错误，可以降低并发数： python translate.py input.srt --concurrency 3

术语表工作原理
程序会采用动态注入策略：

读取整个术语表。

在翻译每一组字幕（Chunk）时，自动检测该片段中出现了哪些术语。

仅将当前片段相关的术语规则放入 Prompt 中。 这样既保证了翻译准确性，又节省了 Token 消耗。

注意事项
API 成本：虽然程序通过合并请求节省了 Token，但处理长视频仍会消耗一定的 API 额度，请注意你的账户余额。

合并逻辑：智能合并功能会修改原始时间轴（将两句的时间合并），如果需要严格对齐原视频口型，请谨慎使用或使用 --no-merge。

License
MIT License
