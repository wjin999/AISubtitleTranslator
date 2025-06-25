#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""translate_srt.py — 单文件运行版 (上下文摘要增强版 v2)
================================================
★ 所有可调参数集中在脚本顶部常量 ★
直接在 VS Code 里 👉 右键 Run Python File 或 F5 即可。
依赖：`pip install openai tqdm`
"""

from __future__ import annotations

import os
import re
import logging
import time
import string # 导入string模块
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Dict, Any

try:
    from openai import OpenAI, OpenAIError, APIConnectionError, RateLimitError # type: ignore
    from tqdm import tqdm # type: ignore
except ImportError as exc:
    if 'openai' in str(exc):
        raise ImportError("请先安装 openai：pip install openai") from exc
    if 'tqdm' in str(exc):
        raise ImportError("请先安装 tqdm：pip install tqdm") from exc
    raise

# ---------------------------------------------------------------------------
# ✏️ 1. 在这里填写你的配置
# ---------------------------------------------------------------------------
INPUT_PATH: str = r"" # ① 输入 SRT 文件
OUTPUT_PATH: str = r""                         # ② 输出文件；留空则自动在原文件名前加上 DEFAULT_TRANSLATED_PREFIX
DEFAULT_TRANSLATED_PREFIX: str = "translated_" # ③ 当OUTPUT_PATH为空时，添加到原文件名前的前缀

# --- API 配置 ---
API_KEY: str = os.environ.get("DEEPSEEK_API_KEY") or "" # ④ DeepSeek API Key（必填）
if API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE":
    logging.warning("请将 'YOUR_DEEPSEEK_API_KEY_HERE' 替换为你的 DeepSeek API Key，或设置 DEEPSEEK_API_KEY 环境变量。")
BASE_URL: str = "https://api.deepseek.com" # DeepSeek API Base URL
MODEL_NAME: str = "deepseek-reasoner" # 主翻译模型
SUMMARY_MODEL_NAME: str = "deepseek-reasoner" # 用于生成摘要的模型

# --- 翻译参数 ---
CHUNK_SIZE_FOR_TRANSLATION: int = 10 # 阶段二翻译时，每次请求的字幕行数 (建议1-10)
MAX_CHARS_PER_ENTRY: int = 300      # 合并后单条字幕最大字符数 (英文)
TEMPERATURE: float = 0.2            # LLM 温度
SUMMARY_MAX_TOKENS: int = 300       # 生成摘要时允许的最大token数

# --- 重试参数 ---
MAX_RETRIES: int = 3
RETRY_DELAY_SECONDS: int = 5
# ---------------------------------------------------------------------------

# 定义需要移除或替换的标点符号集合
# PUNCTS_TO_REMOVE_OR_REPLACE 会在 if __name__ == "__main__": 中定义并赋值
PUNCTS_TO_REMOVE_OR_REPLACE: str = ""
SENTENCE_ENDERS_FOR_MERGE = ".!?…。" # 用于英文原文断句判断的标点


@dataclass
class SrtEntry:
    index: int
    start: str
    end: str
    text: str

    @property
    def timecode(self) -> str:
        return f"{self.start} --> {self.end}"

    def to_srt(self, new_idx: int | None = None) -> str:
        idx = new_idx if new_idx is not None else self.index
        return f"{idx}\n{self.timecode}\n{self.text}\n\n"

def parse_srt(content: str) -> List[SrtEntry]:
    pattern = (
        r"(\d+)\s*\n"
        r"\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
        r"([\s\S]*?)(?=\n\s*\n\s*\d+\s*\n|$)"
    )
    entries: list[SrtEntry] = []
    for idx, start, end, text in re.findall(pattern, content):
        clean = " ".join(line.strip() for line in text.strip().splitlines())
        entries.append(SrtEntry(int(idx), start, end, clean))
    return entries

def should_merge(cur_text: str, next_text: str, max_chars: int) -> bool:
    if not cur_text or not next_text: return False
    if len(cur_text) + 1 + len(next_text) > max_chars: return False
    if cur_text[-1] not in SENTENCE_ENDERS_FOR_MERGE:
        if cur_text[-1] == ',' and next_text and next_text[0].islower(): return True
        if len(cur_text.split()) < 10: return True
    if next_text and next_text[0].islower(): return True
    if re.match(r"^(?:I|I'm|I'll|I'd|I've)\b", next_text, re.IGNORECASE):
        if cur_text[-1] not in SENTENCE_ENDERS_FOR_MERGE: return True
    if len(cur_text.split()) < 4 and cur_text[-1] not in SENTENCE_ENDERS_FOR_MERGE: return True
    if len(next_text.split()) < 3 and cur_text[-1] not in SENTENCE_ENDERS_FOR_MERGE: return True
    return False

def merge_entries(entries: Sequence[SrtEntry], max_chars: int) -> List[SrtEntry]:
    if not entries: return []
    merged: list[SrtEntry] = []
    current_entry = SrtEntry(entries[0].index, entries[0].start, entries[0].end, entries[0].text)
    for i in range(1, len(entries)):
        next_entry = entries[i]
        if should_merge(current_entry.text, next_entry.text, max_chars):
            current_entry.text += " " + next_entry.text
            current_entry.end = next_entry.end
        else:
            merged.append(current_entry)
            current_entry = SrtEntry(next_entry.index, next_entry.start, next_entry.end, next_entry.text)
    merged.append(current_entry)
    return merged

def save_srt(entries: Sequence[SrtEntry], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for new_idx, e in enumerate(entries, 1):
            f.write(e.to_srt(new_idx))

def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]

def process_translated_text_punctuation(text: str) -> str:
    """
    处理翻译后的文本：移除所有在 PUNCTS_TO_REMOVE_OR_REPLACE 中定义的标点符号。
    """
    global PUNCTS_TO_REMOVE_OR_REPLACE # 确保能访问到在main guard中定义的全局变量
    # 1. 移除LLM可能添加的行号或列表标记
    text = re.sub(r'^\s*(\d+[\.\)]\s*|[-*\u2022]\s*)', '', text.strip())

    # 2. 将PUNCTS_TO_REMOVE_OR_REPLACE中定义的标点符号替换为空格
    if PUNCTS_TO_REMOVE_OR_REPLACE: # 确保PUNCTS_TO_REMOVE_OR_REPLACE已初始化
        escaped_puncts = re.escape(PUNCTS_TO_REMOVE_OR_REPLACE)
        text = re.sub(f"[{escaped_puncts}]+", " ", text)

    # 3. 归一化空格并移除首尾空格
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------------------------
# LLM 辅助函数
# ---------------------------------------------------------------------------

def _call_llm_api_with_retry(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int | None = None) -> str:
    """封装LLM API调用，包含重试逻辑"""
    attempt = 0
    last_exception = None
    while attempt < MAX_RETRIES:
        try:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                params["max_tokens"] = max_tokens

            response = client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except RateLimitError as e:
            logging.warning(f"API速率限制：{e}。将在 {RETRY_DELAY_SECONDS} 秒后重试 ({attempt+1}/{MAX_RETRIES})")
            last_exception = e
            time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
        except APIConnectionError as e:
            logging.warning(f"API连接错误：{e}。将在 {RETRY_DELAY_SECONDS * (attempt + 1)} 秒后重试 ({attempt+1}/{MAX_RETRIES})")
            last_exception = e
            time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
        except OpenAIError as e: # 其他 OpenAI 错误
            logging.error(f"OpenAI API 错误：{e} (尝试 {attempt+1}/{MAX_RETRIES})")
            last_exception = e
            time.sleep(RETRY_DELAY_SECONDS)
        except Exception as e: # 其他未知错误
            logging.error(f"LLM调用未知错误：{e} (尝试 {attempt+1}/{MAX_RETRIES})")
            last_exception = e
            time.sleep(RETRY_DELAY_SECONDS)
        attempt += 1
    
    logging.error(f"LLM调用失败，已达最大重试次数。最后错误: {last_exception}")
    return ""


def generate_translation_context_summary(full_text: str, client: OpenAI) -> str:
    """
    使用LLM为整个字幕文本生成一个摘要，作为后续翻译的上下文。
    """
    if not full_text.strip():
        return ""
    
    logging.info("正在生成全文摘要以增强翻译上下文...")
    
    system_prompt = "你是一个文本摘要助手。请仔细阅读以下文本，并生成一个简洁的摘要，概括其主要内容、主题、涉及的关键人物或概念，以及整体的语气和风格。这个摘要将用于辅助后续的字幕翻译。"
    user_prompt = (
        f"请为以下文本内容生成一个摘要（中文，大约 {SUMMARY_MAX_TOKENS // 4} 到 {SUMMARY_MAX_TOKENS // 2} 个汉字）：\n\n"
        "---文本开始---\n"
        f"{full_text}"
        "\n---文本结束---\n\n"
        "摘要应包含：\n"
        "1. 主要讨论的话题或事件。\n"
        "2. 提及的关键实体（如人名、地名、组织名、专业术语等）。\n"
        "3. 文本的整体风格和语境（例如：正式访谈、非正式对话、技术教程、游戏解说等）。"
        "请直接输出摘要内容，不需要其他额外说明。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    summary = _call_llm_api_with_retry(client, SUMMARY_MODEL_NAME, messages, 0.3, SUMMARY_MAX_TOKENS)
    
    if summary:
        logging.info(f"生成的上下文摘要：{summary}")
    else:
        logging.warning("未能生成上下文摘要。")
    return summary


def translate_texts_with_llm_and_context(
    texts_to_translate: List[str],
    client: OpenAI,
    overall_context_summary: str
) -> List[str]:
    """
    使用LLM翻译文本块，同时提供全局上下文摘要。
    """
    if not API_KEY or API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE":
        logging.error("API Key 未设置或无效。")
        return [f"[翻译跳过，API Key无效] {t}" for t in texts_to_translate]

    translated_results: list[str] = []
    system_prompt_template = (
        "你是一名顶级的专业字幕翻译员和校对员，任务是将英文视频字幕精准、自然地翻译成简体中文。"
        "请严格遵守以下指示进行翻译。"
    )

    # 全局上下文（来自LLM生成的摘要）
    global_context_info = ""
    if overall_context_summary: # 使用 generate_translation_context_summary 生成的摘要
        global_context_info += f"\n\n# 全文摘要和关键点 (请在翻译时参考此上下文)：\n{overall_context_summary}"


    chunks = chunk_list(texts_to_translate, CHUNK_SIZE_FOR_TRANSLATION)
    for chunk_idx, current_chunk in enumerate(tqdm(chunks, desc="翻译字幕块")):
        
        numbered_source_texts = [f"原文{i+1}: {text_item}" for i, text_item in enumerate(current_chunk)]

        user_prompt = (
            f"{global_context_info}\n\n"
            "# 当前待翻译的字幕段落：\n"
            f"以下英文文本（标记为“原文X:”）源自语音识别，可能包含不准确的单词、口语化表达或不自然的断句。\n"
            "翻译要求：\n"
            "1. 仔细阅读每一行“原文X:”，结合上面提供的“全文摘要和关键点”来理解其在整个对话中的含义。\n"
            "2. 修正明显的语音识别错误，并调整不自然的断句，使其符合中文的自然表达。\n"
            "3. 如果摘要中提到了关键术语或人名，请尽量保持翻译的一致性。\n"
            "4. 确保译文准确、流畅、地道，并保持原文的语气风格。\n"
            "5. **输出格式**：对于每一行“原文X:”，请给出对应的翻译，并以“译文X:”开头。例如，若原文是“原文1: Hello world”，则输出“译文1: 你好 世界”。\n"
            "   **请确保翻译的行数与原文的行数完全一致。不要合并或拆分行。**\n\n"
            "------待翻译原文开始------\n"
            + "\n".join(numbered_source_texts) +
            "\n------待翻译原文结束------\n"
            "请仅输出“译文X:”格式的翻译结果，每行一个，不要包含任何额外的解释或说明。"
        )

        messages = [
            {"role": "system", "content": system_prompt_template},
            {"role": "user", "content": user_prompt},
        ]
        
        llm_output = _call_llm_api_with_retry(client, MODEL_NAME, messages, TEMPERATURE)

        parsed_lines = ["[翻译获取失败或格式错误]" for _ in range(len(current_chunk))] # 初始化占位符
        
        if llm_output:
            raw_api_lines = llm_output.splitlines()
            parsed_count = 0
            temp_parsed_lines = {} # 使用字典存储，key为行号

            for line_text in raw_api_lines:
                line_text = line_text.strip()
                if not line_text: continue

                match = re.match(r"^\s*(?:译文|TR_LINE_)\s*(\d+)\s*[:：\s]+\s*(.*)", line_text, re.IGNORECASE)
                if match:
                    try:
                        idx = int(match.group(1)) - 1 # 0-based
                        translation = match.group(2).strip()
                        if 0 <= idx < len(current_chunk):
                            if parsed_lines[idx] == "[翻译获取失败或格式错误]": # 确保只填充一次
                                parsed_lines[idx] = translation
                                parsed_count += 1
                        else:
                            logging.warning(f"块 {chunk_idx+1}: 解析到无效的译文行号 {idx+1} (超出范围 {len(current_chunk)})，内容: '{line_text}'")
                    except ValueError:
                        logging.warning(f"块 {chunk_idx+1}: 解析译文行号失败: '{line_text}'")
                else:
                    logging.debug(f"块 {chunk_idx+1}: 收到未按预期格式的行: '{line_text}' (将尝试作为非标记行处理)")
            
            # 如果严格按标记解析后数量不足，并且LLM返回的非标记行数恰好能补足，则尝试使用
            if parsed_count < len(current_chunk):
                non_marked_lines = [l.strip() for l in raw_api_lines if l.strip() and not re.match(r"^\s*(?:译文|TR_LINE_)\s*\d+\s*[:：\s]+", l, re.IGNORECASE)]
                if len(non_marked_lines) == len(current_chunk) and parsed_count == 0: # 如果完全没有TR_LINE标记，但行数匹配
                    logging.warning(f"块 {chunk_idx+1}: LLM未按TR_LINE_X格式输出，但行数匹配，直接使用输出。")
                    parsed_lines = non_marked_lines
                elif parsed_count < len(current_chunk): # 仍然不足，补齐
                     logging.warning(
                        f"块 {chunk_idx+1}/{len(chunks)}：最终解析得到的翻译行数 ({parsed_count}) 少于原文行数 ({len(current_chunk)})。"
                        f"将对缺失部分使用占位符。"
                    )
                    # 对于未被填充的行，保持为占位符

            translated_results.extend(parsed_lines)
        else: # llm_output为空
            logging.error(f"块 {chunk_idx+1} LLM API调用返回空内容。")
            translated_results.extend([f"[翻译API空返回] {text_item}" for text_item in current_chunk])

    return translated_results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    global PUNCTS_TO_REMOVE_OR_REPLACE # 声明我们要修改全局变量

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

    # 定义更全面的标点符号集，用于最终的字幕清理
    _PUNCT_ASCII = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    _PUNCT_CHINESE_AND_FULLWIDTH = (
        "、 。 ， ？ ！ ： ； （ ） 【 】 「 」 『 』 « » ‹ › 〈 〉 《 》 “ ” ‘ ’ „ ‟ ·"  # 中文常用及全角标点
        "…"  # 省略号 (U+2026)
        "— – ― ─"  # 各种破折号
        "～"  # 波浪号
        # 添加用户明确指出的全角符号的 Unicode 形式，以防万一
        "\u3001"  # 、 Dunhao
        "\uff0c"  # ， Fullwidth Comma
        "\u3002"  # 。 Ideographic Full Stop
        "\uff1f"  # ？ Fullwidth Question Mark
        "\uff01"  # ！ Fullwidth Exclamation Mark
        "\uff1a"  # ： Fullwidth Colon
        "\uff1b"  # ； Fullwidth Semicolon
        "\uff08"  # （ Fullwidth Left Parenthesis
        "\uff09"  # ） Fullwidth Right Parenthesis
        "\u300a"  # 《
        "\u300b"  # 》
        "\u201c"  # “
        "\u201d"  # ”
        "\u2018"  # ‘
        "\u2019"  # ’
    )
    # 合并所有标点符号，并去重
    combined_puncts = set(_PUNCT_ASCII)
    for char in _PUNCT_CHINESE_AND_FULLWIDTH:
        combined_puncts.add(char)
    PUNCTS_TO_REMOVE_OR_REPLACE = "".join(sorted(list(combined_puncts)))
    logging.debug(f"用于移除的标点符号集: {PUNCTS_TO_REMOVE_OR_REPLACE}")


    if not API_KEY or API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE":
        logging.error("API Key 未配置。请设置 API_KEY 或 DEEPSEEK_API_KEY 环境变量。")
        return

    in_path = Path(INPUT_PATH).expanduser().resolve()
    if not in_path.is_file():
        logging.error(f"输入文件未找到: {in_path}")
        raise FileNotFoundError(in_path)

    if OUTPUT_PATH:
        out_path = Path(OUTPUT_PATH).expanduser().resolve()
        logging.info(f"输出文件路径已指定: {out_path}")
    else:
        out_path = in_path.with_name(f"{DEFAULT_TRANSLATED_PREFIX}{in_path.name}")
        logging.info(f"输出文件路径 (OUTPUT_PATH) 为空，将自动为原文件名添加前缀 '{DEFAULT_TRANSLATED_PREFIX}' 生成新文件: {out_path}")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"读取原始 SRT: {in_path}")
    try:
        original_srt_content = in_path.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"读取 SRT 文件失败: {e}"); return
        
    entries = parse_srt(original_srt_content)
    logging.info(f"解析到 {len(entries)} 条原始字幕")
    if not entries: logging.warning("SRT 文件中未解析到任何字幕条目。"); return

    merged_entries = merge_entries(entries, MAX_CHARS_PER_ENTRY)
    logging.info(f"合并短句后剩余 {len(merged_entries)} 条字幕预备翻译")
    if not merged_entries: logging.warning("合并后没有字幕条目可供翻译。"); return

    english_texts = [entry.text for entry in merged_entries]
    
    full_english_text_for_summary = "\n".join(english_texts)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    overall_context_summary = ""
    if full_english_text_for_summary.strip():
        overall_context_summary = generate_translation_context_summary(full_english_text_for_summary, client)
    else:
        logging.info("没有足够的文本内容来生成摘要。")

    logging.info(f"开始使用模型 {MODEL_NAME} 进行翻译 (每块 {CHUNK_SIZE_FOR_TRANSLATION} 条，带上下文摘要)...")
    
    translated_chinese_texts = translate_texts_with_llm_and_context(
        english_texts,
        client,
        overall_context_summary
    )

    if len(translated_chinese_texts) != len(merged_entries):
        logging.error(
            f"最终翻译文本数 ({len(translated_chinese_texts)}) 与合并字幕条目数 ({len(merged_entries)}) 不匹配。"
        )
        min_len = min(len(translated_chinese_texts), len(merged_entries))
        merged_entries = merged_entries[:min_len]
        translated_chinese_texts = translated_chinese_texts[:min_len]

    final_translated_entries: list[SrtEntry] = []
    for original_entry, translated_text in zip(merged_entries, translated_chinese_texts):
        cleaned_translated_text = process_translated_text_punctuation(translated_text)
        final_translated_entries.append(
            SrtEntry(
                index=original_entry.index,
                start=original_entry.start,
                end=original_entry.end,
                text=cleaned_translated_text,
            )
        )
    
    if not final_translated_entries:
        logging.warning("没有生成任何最终翻译条目。"); return

    save_srt(final_translated_entries, out_path)
    logging.info(f"翻译完成 → {out_path}")


if __name__ == "__main__":
    main()