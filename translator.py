#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import logging
import time
import string
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Dict, Any

try:
    from openai import OpenAI, OpenAIError, APIConnectionError, RateLimitError
    from tqdm import tqdm
except ImportError as exc:
    raise ImportError(f"Required package not found: {exc}. Please run 'pip install -r requirements.txt'")

# --- Global constants that will be configured via argparse ---
DEFAULT_TRANSLATED_PREFIX: str = "translated_"
SENTENCE_ENDERS_FOR_MERGE = ".!?…。"
PUNCTS_TO_REMOVE_OR_REPLACE: str = ""

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
    global PUNCTS_TO_REMOVE_OR_REPLACE
    text = re.sub(r'^\s*(\d+[\.\)]\s*|[-*\u2022]\s*)', '', text.strip())
    if PUNCTS_TO_REMOVE_OR_REPLACE:
        escaped_puncts = re.escape(PUNCTS_TO_REMOVE_OR_REPLACE)
        text = re.sub(f"[{escaped_puncts}]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _call_llm_api_with_retry(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int | None, max_retries: int, retry_delay: int) -> str:
    attempt = 0
    last_exception = None
    while attempt < max_retries:
        try:
            params = {"model": model, "messages": messages, "temperature": temperature}
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            response = client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except (RateLimitError, APIConnectionError, OpenAIError) as e:
            logging.warning(f"API Error ({type(e).__name__}): {e}. Retrying in {retry_delay}s... ({attempt+1}/{max_retries})")
            last_exception = e
            time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e} (Attempt {attempt+1}/{max_retries})")
            last_exception = e
            time.sleep(retry_delay)
        attempt += 1
    logging.error(f"LLM call failed after {max_retries} retries. Last error: {last_exception}")
    return ""

def generate_translation_context_summary(full_text: str, client: OpenAI, model: str, max_tokens: int, max_retries: int, retry_delay: int) -> str:
    if not full_text.strip(): return ""
    logging.info("Generating a summary of the full text to enhance translation context...")
    system_prompt = "你是一个文本摘要助手。请仔细阅读以下文本，并生成一个简洁的摘要，概括其主要内容、主题、涉及的关键人物或概念，以及整体的语气和风格。这个摘要将用于辅助后续的字幕翻译。"
    user_prompt = f"请为以下文本内容生成一个摘要（中文，大约 {max_tokens // 4} 到 {max_tokens // 2} 个汉字）：\n\n---文本开始---\n{full_text}\n---文本结束---\n\n摘要应包含：\n1. 主要讨论的话题或事件。\n2. 提及的关键实体（如人名、地名、组织名、专业术语等）。\n3. 文本的整体风格和语境（例如：正式访谈、非正式对话、技术教程、游戏解说等）。请直接输出摘要内容，不需要其他额外说明。"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    summary = _call_llm_api_with_retry(client, model, messages, 0.3, max_tokens, max_retries, retry_delay)
    if summary:
        logging.info(f"Context summary generated: {summary}")
    else:
        logging.warning("Failed to generate a context summary.")
    return summary

def translate_texts_with_llm_and_context(texts: List[str], client: OpenAI, context_summary: str, model: str, chunk_size: int, temperature: float, max_retries: int, retry_delay: int) -> List[str]:
    translated_results = []
    system_prompt = "你是一名顶级的专业字幕翻译员和校对员，任务是将英文视频字幕精准、自然地翻译成简体中文。请严格遵守以下指示进行翻译。"
    context_info = f"\n\n# 全文摘要和关键点 (请在翻译时参考此上下文)：\n{context_summary}" if context_summary else ""
    
    chunks = chunk_list(texts, chunk_size)
    for i, chunk in enumerate(tqdm(chunks, desc="Translating Subtitle Chunks")):
        numbered_source = [f"原文{j+1}: {text}" for j, text in enumerate(chunk)]
        user_prompt = (
            f"{context_info}\n\n"
            "# 当前待翻译的字幕段落：\n"
            "以下英文文本（标记为“原文X:”）源自语音识别，可能包含不准确的单词、口语化表达或不自然的断句。\n"
            "翻译要求：\n"
            "1. 仔细阅读每一行“原文X:”，结合上面提供的“全文摘要和关键点”来理解其在整个对话中的含义。\n"
            "2. 修正明显的语音识别错误，并调整不自然的断句，使其符合中文的自然表达。\n"
            "3. 如果摘要中提到了关键术语或人名，请尽量保持翻译的一致性。\n"
            "4. 确保译文准确、流畅、地道，并保持原文的语气风格。\n"
            "5. **输出格式**：对于每一行“原文X:”，请给出对应的翻译，并以“译文X:”开头。例如，若原文是“原文1: Hello world”，则输出“译文1: 你好 世界”。\n"
            "   **请确保翻译的行数与原文的行数完全一致。不要合并或拆分行。**\n\n"
            "------待翻译原文开始------\n" + "\n".join(numbered_source) + "\n------待翻译原文结束------\n"
            "请仅输出“译文X:”格式的翻译结果，每行一个，不要包含任何额外的解释或说明。"
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        llm_output = _call_llm_api_with_retry(client, model, messages, temperature, None, max_retries, retry_delay)
        
        parsed_lines = ["[Translation failed or format error]" for _ in chunk]
        if llm_output:
            found_translations = {int(m.group(1)) - 1: m.group(2).strip() for m in re.finditer(r"^\s*(?:Translation|译文)\s*(\d+)\s*[:：\s]+(.*)", llm_output, re.MULTILINE | re.IGNORECASE)}
            for j in range(len(chunk)):
                if j in found_translations:
                    parsed_lines[j] = found_translations[j]
                else:
                    logging.warning(f"Chunk {i+1}: Could not find translation for line {j+1}. Using placeholder.")
        else:
            logging.error(f"Chunk {i+1}: LLM API call returned empty content.")
            
        translated_results.extend(parsed_lines)
    return translated_results

def main_flow(args):
    global PUNCTS_TO_REMOVE_OR_REPLACE
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

    _PUNCT_ASCII = string.punctuation
    _PUNCT_CHINESE = "、。，？！：；（）【】「」『』«»‹›〈〉《》“”…—–―─～·"
    combined_puncts = set(_PUNCT_ASCII + _PUNCT_CHINESE)
    PUNCTS_TO_REMOVE_OR_REPLACE = "".join(sorted(list(combined_puncts)))

    if not args.api_key:
        logging.error("API Key is not configured. Use the --api-key argument or set the DEEPSEEK_API_KEY environment variable.")
        return

    in_path = Path(args.input_path).expanduser().resolve()
    if not in_path.is_file():
        logging.error(f"Input file not found: {in_path}")
        return

    if args.output_path:
        out_path = Path(args.output_path).expanduser().resolve()
    else:
        out_path = in_path.with_name(f"{DEFAULT_TRANSLATED_PREFIX}{in_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading source SRT: {in_path}")
    try:
        original_content = in_path.read_text(encoding="utf-8-sig")
    except Exception as e:
        logging.error(f"Failed to read SRT file: {e}")
        return

    entries = parse_srt(original_content)
    logging.info(f"Parsed {len(entries)} original subtitle entries.")
    if not entries: 
        logging.warning("No subtitle entries found in the SRT file.")
        return

    merged_entries = merge_entries(entries, args.max_chars_per_entry)
    logging.info(f"Merged into {len(merged_entries)} entries for translation.")
    if not merged_entries: 
        logging.warning("No entries left to translate after merging.")
        return
    
    english_texts = [entry.text for entry in merged_entries]
    full_english_text = "\n".join(english_texts)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    
    context_summary = generate_translation_context_summary(full_english_text, client, args.summary_model_name, args.summary_max_tokens, args.max_retries, args.retry_delay_seconds)
    
    logging.info(f"Starting translation with model {args.model_name}...")
    translated_texts = translate_texts_with_llm_and_context(english_texts, client, context_summary, args.model_name, args.chunk_size_for_translation, args.temperature, args.max_retries, args.retry_delay_seconds)

    if len(translated_texts) != len(merged_entries):
        logging.error(f"Mismatch between translated text count ({len(translated_texts)}) and source entry count ({len(merged_entries)}). Truncating to the shorter length.")
        min_len = min(len(translated_texts), len(merged_entries))
        merged_entries, translated_texts = merged_entries[:min_len], translated_texts[:min_len]

    final_entries = [SrtEntry(orig.index, orig.start, orig.end, process_translated_text_punctuation(trans)) for orig, trans in zip(merged_entries, translated_texts)]
    
    if not final_entries:
        logging.warning("No final translated entries were generated.")
        return

    save_srt(final_entries, out_path)
    logging.info(f"Translation complete. Output saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate SRT subtitle files using a Large Language Model API.")
    parser.add_argument("input_path", help="Path to the source SRT file.")
    parser.add_argument("output_path", nargs='?', default=None, help="Path to the output SRT file. If omitted, adds 'translated_' prefix to the original filename.")
    
    api_group = parser.add_argument_group('API Configuration')
    api_group.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"), help="API Key for the LLM service. Can also be set via DEEPSEEK_API_KEY environment variable.")
    api_group.add_argument("--base-url", default="https://api.deepseek.com", help="The base URL for the API endpoint.")
    
    model_group = parser.add_argument_group('Model and Translation Parameters')
    model_group.add_argument("--model", dest="model_name", default="deepseek-reasoner", help="LLM model for the main translation task.")
    model_group.add_argument("--summary-model", dest="summary_model_name", default="deepseek-reasoner", help="LLM model for the initial summarization task.")
    model_group.add_argument("--chunk-size", dest="chunk_size_for_translation", type=int, default=10, help="Number of subtitle lines to translate per API call.")
    model_group.add_argument("--max-chars", dest="max_chars_per_entry", type=int, default=300, help="Maximum character length for a single merged subtitle entry before translation.")
    model_group.add_argument("--temperature", type=float, default=0.2, help="LLM generation temperature (e.g., 0.2 for more deterministic output).")
    model_group.add_argument("--summary-max-tokens", type=int, default=300, help="Maximum tokens for the context summary generation.")
    
    retry_group = parser.add_argument_group('Retry Logic')
    retry_group.add_argument("--retries", dest="max_retries", type=int, default=3, help="Maximum number of retries for a failed API call.")
    retry_group.add_argument("--retry-delay", dest="retry_delay_seconds", type=int, default=5, help="Delay in seconds between retries.")
    
    args = parser.parse_args()
    if not args.api_key:
         # A check for api key in arguments as environment variable may not be set.
         # This should not be a problem in the Canvas environment, but is a good practice for standalone scripts.
        parser.error("API key must be provided via --api-key argument or DEEPSEEK_API_KEY environment variable.")
    main_flow(args)
