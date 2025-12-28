"""Core translation logic using LLM."""

from __future__ import annotations

import asyncio
import json
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from openai import AsyncOpenAI

from .llm_client import call_llm_async
from .glossary import find_matching_terms, Glossary
from .text_utils import validate_translation

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """单条翻译结果。"""
    index: int
    original: str
    translated: str
    success: bool
    error: str = ""


async def generate_context_summary(
    full_text: str, 
    client: AsyncOpenAI, 
    model: str
) -> str:
    """
    Generate a context summary for the subtitle content.
    """
    if not full_text.strip():
        return ""
    
    logger.info(f"Generating context summary using model: {model}...")
    
    system_prompt = (
        "You are a professional content analyst. "
        "Read the text and generate a concise background summary including: "
        "main topics, key terms, character relationships, and overall tone. "
        "Output in Chinese, within 150 words."
    )
    
    # 截断过长文本
    max_len = 6000
    if len(full_text) > max_len:
        # 取开头和结尾
        truncated = full_text[:max_len//2] + "\n...\n" + full_text[-max_len//2:]
    else:
        truncated = full_text
    
    content = await call_llm_async(
        client, 
        model, 
        [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": truncated}
        ], 
        temperature=0.3,
        max_retries=2
    )
    
    if content:
        logger.info(f"Context summary generated ({len(content)} chars)")
    else:
        logger.warning("Failed to generate context summary")
    
    return content


def _build_translation_prompt(
    items: List[Dict[str, Any]],
    context_prev: List[str],
    context_next: List[str],
    global_summary: str,
    matched_terms: List[str]
) -> tuple[str, str]:
    """Build translation prompts."""
    
    glossary_section = ""
    if matched_terms:
        glossary_list = "\n".join([f"  - {t}" for t in matched_terms])
        glossary_section = f"\n## Glossary (must use):\n{glossary_list}\n"
    
    system_prompt = """You are a professional subtitle translator. Translate English to Simplified Chinese.

## Rules:
1. Output valid JSON: {"translations": [{"id": 0, "text": "翻译"}, ...]}
2. Keep the same number of items as input
3. Keep translations concise for subtitles
4. Use natural Chinese expressions
5. Follow glossary terms exactly if provided

## JSON Format Example:
{"translations": [{"id": 0, "text": "你好"}, {"id": 1, "text": "世界"}]}"""

    prev_str = " | ".join(context_prev[-3:]) if context_prev else ""
    next_str = " | ".join(context_next[:3]) if context_next else ""
    
    context_section = ""
    if prev_str or next_str:
        context_section = f"\n## Context:\nPrev: {prev_str or 'N/A'}\nNext: {next_str or 'N/A'}\n"
    
    summary_section = ""
    if global_summary:
        summary_section = f"\n## Background:\n{global_summary[:300]}\n"
    
    user_prompt = f"""{summary_section}{glossary_section}{context_section}
## Translate:
{json.dumps(items, ensure_ascii=False)}

Output JSON only:"""
    
    return system_prompt, user_prompt


def _parse_translation_response(json_str: str, expected_count: int) -> Dict[int, str]:
    """Parse JSON response from translation API."""
    
    if not json_str:
        return {}
    
    translated_map: Dict[int, str] = {}
    
    try:
        # 清理可能的 markdown 格式
        clean = json_str.strip()
        clean = re.sub(r'^```(?:json)?\s*', '', clean)
        clean = re.sub(r'\s*```$', '', clean)
        
        data = json.loads(clean)
        
        translations = data.get("translations", [])
        if not isinstance(translations, list):
            logger.warning("'translations' is not a list")
            return {}
        
        for item in translations:
            if not isinstance(item, dict):
                continue
            
            item_id = item.get("id")
            text = item.get("text", "")
            
            # 验证 id 在有效范围内
            if isinstance(item_id, int) and 0 <= item_id < expected_count:
                translated_map[item_id] = str(text)
            else:
                logger.debug(f"Invalid id: {item_id}")
                
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}")
        logger.debug(f"Raw response: {json_str[:200]}...")
    
    return translated_map


async def _translate_single_retry(
    client: AsyncOpenAI,
    model: str,
    original: str,
    global_summary: str,
    glossary: Glossary | Dict[str, str]
) -> str:
    """单条翻译重试（用于 chunk 翻译失败的条目）。"""
    
    matched = find_matching_terms(glossary, original)
    terms = [f"{k} -> {v}" for k, v in matched.items()]
    
    system_prompt = (
        "Translate English to Chinese. "
        "Output only the translation, no explanation."
    )
    
    glossary_hint = f" Terms: {', '.join(terms)}" if terms else ""
    user_prompt = f"Translate:{glossary_hint}\n{original}"
    
    result = await call_llm_async(
        client, model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_retries=2,
        json_mode=False
    )
    
    return result.strip() if result else ""


async def translate_chunk_task(
    client: AsyncOpenAI, 
    chunk_data: List[Dict[str, Any]], 
    context_prev: List[str],
    context_next: List[str],
    global_summary: str,
    glossary: Glossary | Dict[str, str],
    model: str, 
    sem: asyncio.Semaphore,
    retry_failed: bool = True
) -> List[TranslationResult]:
    """
    Translate a chunk of subtitle entries.
    
    Returns:
        List of TranslationResult objects
    """
    async with sem:
        items = [
            {"id": i, "text": item['text']} 
            for i, item in enumerate(chunk_data)
        ]
        
        # 动态术语匹配
        chunk_text = " ".join([item['text'] for item in chunk_data])
        matched = find_matching_terms(glossary, chunk_text)
        matched_terms = [f"{term} -> {trans}" for term, trans in matched.items()]
        
        system_prompt, user_prompt = _build_translation_prompt(
            items, context_prev, context_next, global_summary, matched_terms
        )
        
        json_str = await call_llm_async(
            client, model,
            [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ], 
            temperature=0.3,
            max_retries=3,
            json_mode=True
        )
        
        translated_map = _parse_translation_response(json_str, len(chunk_data))
        
        # 构建结果
        results: List[TranslationResult] = []
        failed_indices: List[int] = []
        
        for i, item in enumerate(chunk_data):
            original = item['text']
            
            if i in translated_map:
                translated = translated_map[i]
                is_valid, error = validate_translation(original, translated)
                
                if is_valid:
                    results.append(TranslationResult(
                        index=item['index'],
                        original=original,
                        translated=translated,
                        success=True
                    ))
                else:
                    logger.debug(f"Validation failed for #{i}: {error}")
                    failed_indices.append(i)
                    results.append(TranslationResult(
                        index=item['index'],
                        original=original,
                        translated="",
                        success=False,
                        error=error
                    ))
            else:
                failed_indices.append(i)
                results.append(TranslationResult(
                    index=item['index'],
                    original=original,
                    translated="",
                    success=False,
                    error="Missing from response"
                ))
        
        # 单条重试失败的项
        if retry_failed and failed_indices:
            logger.info(f"Retrying {len(failed_indices)} failed items individually...")
            
            for i in failed_indices:
                original = chunk_data[i]['text']
                retried = await _translate_single_retry(
                    client, model, original, global_summary, glossary
                )
                
                if retried:
                    is_valid, _ = validate_translation(original, retried)
                    if is_valid:
                        results[i] = TranslationResult(
                            index=chunk_data[i]['index'],
                            original=original,
                            translated=retried,
                            success=True
                        )
                        logger.debug(f"Retry succeeded for #{i}")
        
        return results


def extract_translations(results: List[TranslationResult]) -> List[str]:
    """
    从 TranslationResult 列表中提取翻译文本。
    
    对于失败的项，返回原文（而非错误标记）。
    """
    translations: List[str] = []
    
    for r in results:
        if r.success and r.translated:
            translations.append(r.translated)
        else:
            # 失败时返回原文，让用户知道哪些没翻译
            translations.append(r.original)
    
    return translations
