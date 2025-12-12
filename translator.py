#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
import logging
import asyncio
import argparse
import string
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Dict, Any, Optional

# --- 依赖库检查 ---
try:
    from openai import AsyncOpenAI, APIConnectionError, RateLimitError
    from tqdm.asyncio import tqdm_asyncio
    from dotenv import load_dotenv 
except ImportError as exc:
    raise ImportError(f"缺失必要库: {exc}. 请运行 'pip install openai tqdm python-dotenv'")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

load_dotenv()

# --- 全局变量 ---
DEFAULT_TRANSLATED_PREFIX: str = "translated_"
PUNCTS_TO_REMOVE_OR_REPLACE: str = ""
NLP_MODEL = None

@dataclass
class SrtEntry:
    index: int
    start: str
    end: str
    text: str

    @property
    def timecode(self) -> str:
        return f"{self.start} --> {self.end}"
    
    @property
    def start_seconds(self) -> float:
        return self._time_str_to_seconds(self.start)

    @property
    def end_seconds(self) -> float:
        return self._time_str_to_seconds(self.end)

    @staticmethod
    def _time_str_to_seconds(t_str: str) -> float:
        try:
            h, m, s_full = t_str.split(':')
            s, ms = s_full.split(',')
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        except ValueError:
            return 0.0

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

# --- 新增功能：加载术语表 ---
def load_glossary(path: Path) -> Dict[str, str]:
    glossary = {}
    if not path.exists():
        logging.warning(f"Glossary file not found: {path}")
        return glossary
    
    try:
        content = path.read_text(encoding='utf-8')
        for line in content.splitlines():
            if '=' in line:
                # 支持 "Term = Translation" 格式
                term, trans = line.split('=', 1)
                glossary[term.strip()] = trans.strip()
    except Exception as e:
        logging.error(f"Error loading glossary: {e}")
    
    logging.info(f"Loaded {len(glossary)} terms from glossary.")
    return glossary

# --- spaCy 智能合并 ---
def init_spacy_model():
    global NLP_MODEL
    if not SPACY_AVAILABLE:
        raise ImportError("需要合并字幕但未找到 spaCy。请运行 'pip install spacy' 或使用 --no-merge 参数。")

    logging.info("Initializing spaCy NLP model...")
    try:
        NLP_MODEL = spacy.load("en_core_web_sm", disable=["ner", "textcat", "lemmatizer"])
    except OSError:
        logging.warning("Model 'en_core_web_sm' not found. Downloading...")
        from spacy.cli import download
        download("en_core_web_sm")
        NLP_MODEL = spacy.load("en_core_web_sm", disable=["ner", "textcat", "lemmatizer"])

def should_merge_spacy(cur_entry: SrtEntry, next_entry: SrtEntry, max_chars: int, time_gap_threshold: float) -> bool:
    cur_text = cur_entry.text
    next_text = next_entry.text
    if not cur_text or not next_text: return False
    if len(cur_text) + 1 + len(next_text) > max_chars: return False
    
    time_gap = next_entry.start_seconds - cur_entry.end_seconds
    if time_gap > time_gap_threshold:
        return False

    combined_text = f"{cur_text} {next_text}"
    doc = NLP_MODEL(combined_text)
    sentences = list(doc.sents)
    
    if len(sentences) == 1:
        return True
    
    split_index = len(cur_text) + 1
    for sent in sentences:
        if abs(sent.start_char - split_index) <= 2:
            return False
    return True

def merge_entries(entries: Sequence[SrtEntry], max_chars: int, time_gap_threshold: float) -> List[SrtEntry]:
    if not entries: return []
    merged: list[SrtEntry] = []
    current = entries[0]
    for i in range(1, len(entries)):
        next_entry = entries[i]
        if should_merge_spacy(current, next_entry, max_chars, time_gap_threshold):
            current.text += " " + next_entry.text
            current.end = next_entry.end
        else:
            merged.append(current)
            current = next_entry
    merged.append(current)
    return merged

def save_srt(entries: Sequence[SrtEntry], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for new_idx, e in enumerate(entries, 1):
            f.write(e.to_srt(new_idx))

def process_translated_text_punctuation(text: str) -> str:
    global PUNCTS_TO_REMOVE_OR_REPLACE
    text = re.sub(r'^\s*(\d+[\.:\)]\s*|[-*\u2022]\s*)', '', text.strip())
    if PUNCTS_TO_REMOVE_OR_REPLACE:
        escaped_puncts = re.escape(PUNCTS_TO_REMOVE_OR_REPLACE)
        text = re.sub(f"[{escaped_puncts}]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- 异步 LLM 调用 ---
async def _call_llm_async(
    client: AsyncOpenAI, 
    model: str, 
    messages: List[Dict[str, str]], 
    temperature: float, 
    max_retries: int,
    json_mode: bool = False
) -> str:
    attempt = 0
    while attempt < max_retries:
        try:
            params = {
                "model": model, 
                "messages": messages, 
                "temperature": temperature
            }
            if json_mode:
                params["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except (RateLimitError, APIConnectionError) as e:
            delay = 2 * (attempt + 1)
            logging.warning(f"API Error: {e}. Retrying in {delay}s... ({attempt+1}/{max_retries})")
            await asyncio.sleep(delay)
        except Exception as e:
            logging.error(f"Unexpected Error: {e}")
            await asyncio.sleep(2)
        attempt += 1
    return ""

# --- 摘要生成 ---
async def generate_context_summary(full_text: str, client: AsyncOpenAI, model: str) -> str:
    if not full_text.strip(): return ""
    logging.info(f"Generating context summary using reasoning model: {model}...")
    
    system_prompt = "You are a professional content analyst. Please read the text and generate a concise background summary, including: main topics, key terms, character relationships, and overall tone."
    truncated_text = full_text[:8000] + "\n...(truncated)..." if len(full_text) > 8000 else full_text
    user_prompt = f"Text Content:\n{truncated_text}\n\nPlease output a summary in Chinese (within 200 words)."
    
    content = await _call_llm_async(client, model, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], 0.3, 3)
    if content:
        logging.info(f"Context Summary: {content}")
    return content

# --- 翻译 Worker ---
async def translate_chunk_task(
    client: AsyncOpenAI, 
    chunk_data: List[Dict], 
    context_prev: List[str],
    context_next: List[str],
    global_summary: str,
    glossary: Dict[str, str], # 传入完整的术语表
    model: str, 
    sem: asyncio.Semaphore
) -> List[str]:
    
    async with sem: 
        items_to_translate = [{"id": i, "original": item['text']} for i, item in enumerate(chunk_data)]
        
        # --- 动态术语匹配 (Dynamic Glossary Injection) ---
        # 1. 把待翻译的所有文本拼起来便于搜索
        chunk_text_combined = " ".join([item['text'] for item in chunk_data]).lower()
        
        # 2. 筛选：只有当术语在当前文本中出现时，才加入 Prompt
        matched_terms = []
        for term, trans in glossary.items():
            if term.lower() in chunk_text_combined:
                matched_terms.append(f"{term} -> {trans}")
        
        # 3. 构造术语提示词
        glossary_prompt_section = ""
        if matched_terms:
            glossary_list_str = "\n".join([f"- {t}" for t in matched_terms])
            glossary_prompt_section = f"\n# MANDATORY TERMINOLOGY (Use these translations):\n{glossary_list_str}\n"

        system_prompt = (
            "Role: Professional Subtitle Translator\n"
            "Target Language: Simplified Chinese (简体中文)\n\n"
            "Constraints:\n"
            "1. Format Strictness: You must output a **strictly valid JSON object** containing a 'translations' list. Each item must have 'id' and 'text'.\n"
            "2. Line-by-Line Correspondence: The number of output items MUST strictly match the number of input items. Do not merge or split lines.\n"
            "3. Length Control: Keep translations concise.\n"
            "4. Terminology: Respect the provided glossary strictly.\n"
            "5. Special Instruction: **Use authentic mainland Chinese internet slang where appropriate.**"
        )
        
        prev_str = "\n".join(context_prev) if context_prev else "(None)"
        next_str = "\n".join(context_next) if context_next else "(None)"
        
        user_prompt = f"""
        # Story Context Summary:
        {global_summary}

        {glossary_prompt_section}

        # Previous Context:
        {prev_str}

        # Next Context:
        {next_str}

        # Input Data (JSON):
        {json.dumps(items_to_translate, ensure_ascii=False)}

        Please translate into JSON:
        """
        
        json_str = await _call_llm_async(client, model, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], 0.3, 3, json_mode=True)
        
        translated_map = {}
        try:
            clean_json = re.sub(r"```json|```", "", json_str).strip()
            data = json.loads(clean_json)
            if "translations" in data and isinstance(data["translations"], list):
                for item in data["translations"]:
                    translated_map[item.get("id")] = item.get("text", "")
            else:
                logging.warning("JSON format valid but key 'translations' missing.")
        except json.JSONDecodeError:
            logging.error(f"JSON parsing failed.")
        
        results = []
        for i in range(len(chunk_data)):
            if i in translated_map:
                results.append(translated_map[i])
            else:
                results.append(f"[Fail] {chunk_data[i]['text']}")
        
        return results

async def main_async_flow(args):
    global PUNCTS_TO_REMOVE_OR_REPLACE
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
    
    _PUNCT_ASCII = string.punctuation
    _PUNCT_CHINESE = "、。，？！：；（）【】「」『』«»‹›〈〉《》“”…—–―─～·"
    PUNCTS_TO_REMOVE_OR_REPLACE = "".join(sorted(list(set(_PUNCT_ASCII + _PUNCT_CHINESE))))

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logging.error("Missing API Key. Please create a .env file or pass --api-key.")
        return

    in_path = Path(args.input_path).expanduser().resolve()
    if not in_path.exists():
        logging.error(f"File not found: {in_path}")
        return

    logging.info(f"Reading: {in_path}")
    original_content = in_path.read_text(encoding="utf-8-sig")
    entries = parse_srt(original_content)
    
    # --- 1. 加载术语表 (智能自动检测版) ---
    glossary = {}
    
    # 逻辑优先级：
    # 1. 如果用户命令行指定了 --glossary，就用指定的。
    # 2. 如果没指定，但当前目录下有 "glossary.txt"，就自动加载。
    
    if args.glossary_path:
        # 情况1: 用户指定了文件
        g_path = Path(args.glossary_path).expanduser().resolve()
        glossary = load_glossary(g_path)
    elif Path("glossary.txt").exists():
        # 情况2: 用户没指定，但程序发现目录下有默认文件
        logging.info("Auto-detected 'glossary.txt' in current directory. Loading...")
        glossary = load_glossary(Path("glossary.txt"))
    else:
        # 情况3: 既没指定，也没找到默认文件
        logging.info("No glossary loaded (glossary.txt not found).")
    
    # --- 2. 决定是否合并 ---
    if args.no_merge:
        logging.info("Merging disabled by user. Proceeding with original entries.")
        merged_entries = entries
    else:
        init_spacy_model()
        logging.info("Merging entries with spaCy logic...")
        merged_entries = merge_entries(entries, args.max_chars_per_entry, args.merge_time_gap)
        logging.info(f"Merged {len(entries)} -> {len(merged_entries)} entries.")

    all_texts = [e.text for e in merged_entries]
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)

    full_text_concat = "\n".join(all_texts)
    summary = await generate_context_summary(full_text_concat, client, args.summary_model_name)

    chunk_size = args.chunk_size_for_translation
    chunks = [all_texts[i : i + chunk_size] for i in range(0, len(all_texts), chunk_size)]
    
    tasks = []
    sem = asyncio.Semaphore(args.concurrency)
    
    logging.info(f"Starting translation (Chunks: {len(chunks)})...")

    CTX_WIN = 2
    for i, chunk in enumerate(chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + len(chunk)
        
        prev_start = max(0, start_idx - CTX_WIN)
        context_prev = all_texts[prev_start : start_idx]
        
        next_end = min(len(all_texts), end_idx + CTX_WIN)
        context_next = all_texts[end_idx : next_end]
        
        chunk_data = [{"index": start_idx + j, "text": text} for j, text in enumerate(chunk)]
        
        task = translate_chunk_task(
            client=client,
            chunk_data=chunk_data,
            context_prev=context_prev,
            context_next=context_next,
            global_summary=summary,
            glossary=glossary, # 传递 glossary 字典
            model=args.model_name,
            sem=sem
        )
        tasks.append(task)

    results_list = await tqdm_asyncio.gather(*tasks, desc="Translating")
    
    final_translated_texts = []
    for res in results_list:
        final_translated_texts.extend(res)

    if len(final_translated_texts) != len(merged_entries):
        logging.warning("Length mismatch warning! Truncating.")
    
    min_len = min(len(final_translated_texts), len(merged_entries))
    final_entries = []
    for i in range(min_len):
        clean_text = process_translated_text_punctuation(final_translated_texts[i])
        orig = merged_entries[i]
        final_entries.append(SrtEntry(orig.index, orig.start, orig.end, clean_text))

    out_path = args.output_path if args.output_path else in_path.with_name(f"{DEFAULT_TRANSLATED_PREFIX}{in_path.name}")
    save_srt(final_entries, Path(out_path))
    logging.info(f"Done! Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async LLM Subtitle Translator Pro (Glossary Support)")
    parser.add_argument("input_path", help="Path to SRT file")
    parser.add_argument("output_path", nargs='?', default=None)
    
    # 新增术语表参数
    parser.add_argument("--glossary", dest="glossary_path", help="Path to glossary file (e.g. glossary.txt, format: Term=Translation)")

    parser.add_argument("--no-merge", action="store_true", help="Disable smart merging (translate line-by-line)")
    parser.add_argument("--api-key", help="Optional override for API Key")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", dest="model_name", default="deepseek-chat")
    parser.add_argument("--summary-model", dest="summary_model_name", default="deepseek-reasoner")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--chunk-size", dest="chunk_size_for_translation", type=int, default=10)
    parser.add_argument("--max-chars", dest="max_chars_per_entry", type=int, default=300)
    parser.add_argument("--merge-gap", dest="merge_time_gap", type=float, default=1.5)

    args = parser.parse_args()
    asyncio.run(main_async_flow(args))
