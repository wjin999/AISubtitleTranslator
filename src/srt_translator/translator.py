"""Core translation logic using LLM."""

from __future__ import annotations

import asyncio
import json
import re
import logging
from typing import List, Dict, Any

from openai import AsyncOpenAI

from .llm_client import call_llm_async
from .glossary import find_matching_terms

logger = logging.getLogger(__name__)


async def generate_context_summary(
    full_text: str, 
    client: AsyncOpenAI, 
    model: str
) -> str:
    """
    Generate a context summary for the subtitle content.
    
    Args:
        full_text: Concatenated subtitle text
        client: AsyncOpenAI client
        model: Model to use for summarization
        
    Returns:
        Summary string in Chinese
    """
    if not full_text.strip():
        return ""
    
    logger.info(f"Generating context summary using model: {model}...")
    
    system_prompt = (
        "You are a professional content analyst. "
        "Please read the text and generate a concise background summary, including: "
        "main topics, key terms, character relationships, and overall tone."
    )
    
    # Truncate if too long
    max_len = 8000
    if len(full_text) > max_len:
        truncated_text = full_text[:max_len] + "\n...(truncated)..."
    else:
        truncated_text = full_text
    
    user_prompt = (
        f"Text Content:\n{truncated_text}\n\n"
        "Please output a summary in Chinese (within 200 words)."
    )
    
    content = await call_llm_async(
        client, 
        model, 
        [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ], 
        temperature=0.3,
        max_retries=3
    )
    
    if content:
        logger.info(f"Context Summary: {content}")
    
    return content


def _build_translation_prompt(
    items_to_translate: List[Dict[str, Any]],
    context_prev: List[str],
    context_next: List[str],
    global_summary: str,
    matched_terms: List[str]
) -> tuple[str, str]:
    """
    Build system and user prompts for translation.
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Build glossary section
    glossary_section = ""
    if matched_terms:
        glossary_list = "\n".join([f"- {t}" for t in matched_terms])
        glossary_section = (
            f"\n# MANDATORY TERMINOLOGY (Use these translations):\n"
            f"{glossary_list}\n"
        )
    
    system_prompt = (
        "Role: Professional Subtitle Translator\n"
        "Target Language: Simplified Chinese (简体中文)\n\n"
        "Constraints:\n"
        "1. Format Strictness: Output a **strictly valid JSON object** "
        "containing a 'translations' list. Each item must have 'id' and 'text'.\n"
        "2. Line-by-Line Correspondence: Output count MUST match input count. "
        "Do not merge or split lines.\n"
        "3. Length Control: Keep translations concise.\n"
        "4. Terminology: Respect the provided glossary strictly.\n"
        "5. Style: **Use authentic mainland Chinese internet slang where appropriate.**"
    )
    
    prev_str = "\n".join(context_prev) if context_prev else "(None)"
    next_str = "\n".join(context_next) if context_next else "(None)"
    
    user_prompt = f"""# Story Context Summary:
{global_summary}
{glossary_section}
# Previous Context:
{prev_str}

# Next Context:
{next_str}

# Input Data (JSON):
{json.dumps(items_to_translate, ensure_ascii=False)}

Please translate into JSON:"""
    
    return system_prompt, user_prompt


def _parse_translation_response(
    json_str: str, 
    chunk_size: int
) -> Dict[int, str]:
    """
    Parse JSON response from translation API.
    
    Args:
        json_str: Raw JSON string from API
        chunk_size: Expected number of translations
        
    Returns:
        Dictionary mapping IDs to translated text
    """
    translated_map: Dict[int, str] = {}
    
    try:
        # Clean potential markdown formatting
        clean_json = re.sub(r"```json|```", "", json_str).strip()
        data = json.loads(clean_json)
        
        if "translations" in data and isinstance(data["translations"], list):
            for item in data["translations"]:
                item_id = item.get("id")
                text = item.get("text", "")
                if item_id is not None:
                    translated_map[item_id] = text
        else:
            logger.warning("JSON valid but 'translations' key missing or invalid.")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
    
    return translated_map


async def translate_chunk_task(
    client: AsyncOpenAI, 
    chunk_data: List[Dict[str, Any]], 
    context_prev: List[str],
    context_next: List[str],
    global_summary: str,
    glossary: Dict[str, str],
    model: str, 
    sem: asyncio.Semaphore
) -> List[str]:
    """
    Translate a chunk of subtitle entries.
    
    Args:
        client: AsyncOpenAI client
        chunk_data: List of dicts with 'index' and 'text' keys
        context_prev: Previous context lines
        context_next: Next context lines
        global_summary: Summary of full content
        glossary: Full glossary dictionary
        model: Model to use for translation
        sem: Semaphore for concurrency control
        
    Returns:
        List of translated texts
    """
    async with sem:
        items_to_translate = [
            {"id": i, "original": item['text']} 
            for i, item in enumerate(chunk_data)
        ]
        
        # Dynamic glossary matching
        chunk_text = " ".join([item['text'] for item in chunk_data])
        matched = find_matching_terms(glossary, chunk_text)
        matched_terms = [f"{term} -> {trans}" for term, trans in matched.items()]
        
        system_prompt, user_prompt = _build_translation_prompt(
            items_to_translate,
            context_prev,
            context_next,
            global_summary,
            matched_terms
        )
        
        json_str = await call_llm_async(
            client, 
            model, 
            [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ], 
            temperature=0.3,
            max_retries=3,
            json_mode=True
        )
        
        translated_map = _parse_translation_response(json_str, len(chunk_data))
        
        # Build results with fallback for failures
        results: List[str] = []
        for i in range(len(chunk_data)):
            if i in translated_map:
                results.append(translated_map[i])
            else:
                results.append(f"[Fail] {chunk_data[i]['text']}")
        
        return results
