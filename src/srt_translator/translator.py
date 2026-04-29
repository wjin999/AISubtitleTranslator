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
    model: str,
    custom_prompt: str | None = None
) -> str:
    """
    Generate a context summary for the subtitle content.
    
    Uses uniform multi-segment sampling to extract representative
    portions from across the full text, ensuring the LLM sees content
    from all parts rather than just head and tail.
    
    Args:
        full_text: The full text to summarize
        client: AsyncOpenAI client
        model: Model name to use
        custom_prompt: Optional custom system prompt to override the default
    """
    if not full_text.strip():
        return ""
    
    logger.info(f"Generating context summary using model: {model}...")
    
    if custom_prompt:
        system_prompt = custom_prompt
    else:
        system_prompt = (
            "你是一名专业的内容分析师。"
            "请阅读以下文本，生成一份简洁的背景摘要，内容包括："
            "主要话题、关键术语、角色关系以及整体基调。"
            "用中文输出，不超过 150 字。"
        )
    
    # 使用均匀多段采样提取代表性内容（每段取完整文本块）
    max_len = 6000
    if len(full_text) > max_len:
        lines = full_text.split('\n')
        num_segments = 5
        segments = []
        max_per_segment = max_len // num_segments  # 每段最多 1200 字符

        if len(lines) >= num_segments * 2:
            for i in range(num_segments):
                start_idx = int(i * len(lines) / num_segments)
                end_idx = int((i + 1) * len(lines) / num_segments)
                segment_text = "\n".join(lines[start_idx:end_idx])
                if len(segment_text) > max_per_segment:
                    segment_text = segment_text[:max_per_segment // 2] + "\n...\n" + segment_text[-max_per_segment // 2:]
                segments.append(segment_text)
        else:
            # 行数较少，全部使用
            segments = lines

        truncated = "\n".join(segments)
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
    matched_terms: List[str],
    custom_prompt: str | None = None,
    translation_memory: Dict[str, str] | None = None,
) -> tuple[str, str]:
    """Build translation prompts.

    Args:
        items: Items to translate.
        context_prev: Previous context text.
        context_next: Next context text.
        global_summary: Background summary.
        matched_terms: Glossary term matches.
        custom_prompt: Optional custom system prompt.
        translation_memory: Dict mapping original text -> translated text
            from previously completed chunks, used for terminology consistency.
    """

    if custom_prompt:
        system_prompt = custom_prompt
    else:
        system_prompt = """你是一名专业的影视字幕翻译专家，负责将英文字幕翻译成简体中文。

## 核心要求：
1. 输出合法 JSON 格式：{"translations": [{"id": 0, "text": "翻译"}, ...]}
2. 输出的条目数量必须与输入完全一致
3. 每条译文必须简洁（中文约 3-5 字符对应一秒屏幕时间）

## 翻译最佳实践：
4. 使用自然、口语化的中文，适合对话场景
5. 保留说话者的语气和情感（愤怒、低语、讽刺、兴奋等）
6. 保持角色语气在全篇字幕中一致
7. 遇到习语、双关语或文化特定内容时，采用意译而非直译

## 字幕标点规范（必须严格遵守）：
8. 句末不加句号（。）：无论是陈述句还是祈使句，字幕结尾一律不写句号（。）。字幕的出现和消失本身就起到了断句的作用。
9. 必须保留问号（？）和叹号（！）：用于准确传达疑问或强烈语气。
10. 句中停顿用空格代替逗号（，）：句子内部的停顿使用空格（半角或全角均可），不使用逗号或顿号。
11. 省略号（……）和破折号（——）：表示话语未说完、被打断或声音拖长时规范使用。
12. 引号（""）和括号（（））：专有名词、画外音或内心独白时正常使用。

## JSON 格式示例：
{"translations": [{"id": 0, "text": "你好"}, {"id": 1, "text": "世界"}]}"""
    
    glossary_section = ""
    if matched_terms:
        glossary_list = "\n".join([f"  - {t}" for t in matched_terms])
        glossary_section = f"\n## 术语表（必须使用）：\n{glossary_list}\n"

    # Translation memory section: show up to 5 previously translated pairs
    memory_section = ""
    if translation_memory:
        # Pick up to 5 representative entries from memory
        memory_items = list(translation_memory.items())[:5]
        memory_lines = "\n".join(
            [f"  {orig} -> {trans}" for orig, trans in memory_items]
        )
        memory_section = f"\n## 翻译记忆（参考已有翻译，保持术语一致）：\n{memory_lines}\n"

    prev_str = " | ".join(context_prev[-3:]) if context_prev else ""
    next_str = " | ".join(context_next[:3]) if context_next else ""
    
    context_section = ""
    if prev_str or next_str:
        context_section = f"\n## 上下文：\n前文：{prev_str or 'N/A'}\n后文：{next_str or 'N/A'}\n"
    
    summary_section = ""
    if global_summary:
        summary_section = f"\n## 背景摘要：\n{global_summary[:300]}\n"
    
    user_prompt = f"""{summary_section}{memory_section}{glossary_section}{context_section}
## 请翻译以下内容：
{json.dumps(items, ensure_ascii=False)}

只输出 JSON，不要其他内容："""
    
    return system_prompt, user_prompt


def _parse_translation_response(json_str: str, expected_count: int) -> Dict[int, str]:
    """Parse JSON response from translation API.

    兼容多种返回格式:
    1. {"translations": [{"id": 0, "text": "..."}, ...]}  (dict with translations key)
    2. [{"id": 0, "text": "..."}, ...]                      (plain array)
    3. JSON 后带有额外文本（Extra data），自动忽略多余内容
    """
    
    if not json_str:
        return {}
    
    translated_map: Dict[int, str] = {}
    
    try:
        # 清理可能的 markdown 格式
        clean = json_str.strip()
        clean = re.sub(r'^```(?:json)?\s*', '', clean)
        clean = re.sub(r'\s*```$', '', clean)
        
        # 使用 raw_decode 处理可能带有额外文本的 JSON
        # 例如模型返回: {"translations":[...]}后面还有文字
        # raw_decode 会只解析第一个完整的 JSON 值，忽略后续内容
        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(clean)
        except json.JSONDecodeError:
            # 如果 raw_decode 也失败，回退到普通 json.loads
            data = json.loads(clean)
        
        # 兼容两种顶层格式
        if isinstance(data, dict):
            translations = data.get("translations", [])
            if not isinstance(translations, list):
                logger.warning("'translations' is not a list")
                return {}
        elif isinstance(data, list):
            # 模型直接返回了纯数组格式
            translations = data
        else:
            logger.warning(f"Unexpected JSON type: {type(data).__name__}")
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
    glossary: Glossary | Dict[str, str],
    context_prev: List[str] | None = None,
    context_next: List[str] | None = None,
) -> str:
    """单条翻译重试（用于 chunk 翻译失败的条目）。
    
    Args:
        client: AsyncOpenAI client.
        model: Model name.
        original: Original text to translate.
        global_summary: Background summary for context.
        glossary: Glossary for terminology.
        context_prev: Previous subtitle texts for context.
        context_next: Next subtitle texts for context.
    """
    
    matched = find_matching_terms(glossary, original)
    terms = [f"{k} -> {v}" for k, v in matched.items()]
    
    system_prompt = (
        "你是一名专业的字幕翻译。"
        "将英文字幕翻译成简体中文。"
        "只输出中文翻译，不要任何解释。"
        "严格遵守字幕标点规范：句末不加句号（。），"
        "保留问号（？）和叹号（！），"
        "句中停顿用空格代替逗号。"
        "保持简洁，适合字幕使用。"
    )
    
    glossary_hint = f" 术语：{', '.join(terms)}" if terms else ""
    
    # Build context-aware user prompt
    context_lines = []
    if global_summary:
        context_lines.append(f"[背景摘要]：{global_summary[:200]}")
    if context_prev:
        prev_text = " | ".join(context_prev[-3:])
        context_lines.append(f"[前文]：{prev_text}")
    if context_next:
        next_text = " | ".join(context_next[:3])
        context_lines.append(f"[后文]：{next_text}")
    
    context_str = "\n".join(context_lines)
    if context_str:
        context_str += "\n"
    
    user_prompt = f"""{context_str}请翻译以下句子：
{original}"""
    
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
    retry_failed: bool = True,
    custom_translation_prompt: str | None = None,
    translation_memory: Dict[str, str] | None = None,
) -> List[TranslationResult]:
    """
    Translate a chunk of subtitle entries.

    Args:
        translation_memory: Dict of original -> translated from previous chunks
            for terminology consistency across chunks.

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
            items, context_prev, context_next, global_summary, matched_terms,
            custom_prompt=custom_translation_prompt,
            translation_memory=translation_memory,
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
        
        # 单条重试失败的项（带上下文）
        if retry_failed and failed_indices:
            logger.info(f"Retrying {len(failed_indices)} failed items individually...")
            
            for i in failed_indices:
                original = chunk_data[i]['text']
                retried = await _translate_single_retry(
                    client, model, original, global_summary, glossary,
                    context_prev=context_prev,
                    context_next=context_next,
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
