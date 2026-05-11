"""Timeline-based subtitle quality checking and correction."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Sequence

from .config import TranslatorConfig
from .llm_client import call_llm_async
from .models import SrtEntry
from .text_utils import clean_translated_text

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """A single corrected subtitle entry and the model's short rationale."""

    index: int
    corrected_text: str
    issue: str = ""


def _overlap_seconds(a: SrtEntry, b: SrtEntry) -> float:
    """Return timeline overlap in seconds."""
    return max(0.0, min(a.end_seconds, b.end_seconds) - max(a.start_seconds, b.start_seconds))


def _nearest_original(original_entries: Sequence[SrtEntry], target: SrtEntry) -> List[SrtEntry]:
    """Fallback alignment for entries that do not overlap exactly."""
    if not original_entries:
        return []
    target_mid = (target.start_seconds + target.end_seconds) / 2
    nearest = min(
        original_entries,
        key=lambda entry: abs(((entry.start_seconds + entry.end_seconds) / 2) - target_mid),
    )
    return [nearest]


def align_originals_by_timeline(
    original_entries: Sequence[SrtEntry],
    translated_entry: SrtEntry,
) -> List[SrtEntry]:
    """Find original subtitles that overlap a translated subtitle's timeline."""
    overlaps = [
        entry for entry in original_entries
        if _overlap_seconds(entry, translated_entry) > 0
    ]
    if overlaps:
        return overlaps
    return _nearest_original(original_entries, translated_entry)


def _parse_quality_response(json_str: str, expected_count: int) -> Dict[int, QualityIssue]:
    """Parse a quality-check JSON response into corrections by local id."""
    if not json_str:
        return {}

    clean = json_str.strip()
    clean = re.sub(r'^```(?:json)?\s*', '', clean)
    clean = re.sub(r'\s*```$', '', clean)

    try:
        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(clean)
        except json.JSONDecodeError:
            data = json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.error("Quality check JSON parse failed: %s", exc)
        logger.debug("Raw response: %s", json_str[:300])
        return {}

    if isinstance(data, dict):
        items = data.get("corrections", data.get("items", []))
    elif isinstance(data, list):
        items = data
    else:
        return {}

    parsed: Dict[int, QualityIssue] = {}
    if not isinstance(items, list):
        return parsed

    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        text = item.get("text", item.get("corrected_text", ""))
        issue = item.get("issue", "")
        if isinstance(item_id, int) and 0 <= item_id < expected_count and isinstance(text, str):
            parsed[item_id] = QualityIssue(item_id, clean_translated_text(text), str(issue))
    return parsed


def _build_quality_prompt(items: List[Dict[str, Any]]) -> tuple[str, str]:
    """Build prompts for timeline-based subtitle QA."""
    system_prompt = """你是一名严格的影视字幕质检与润色专家。

任务：根据时间轴匹配的原字幕，检查译文字幕是否存在漏译、错译、未翻译、语气不自然、术语不一致、字幕过长、标点不符合字幕习惯等问题，并输出修正后的译文。

规则：
1. 只修正译文文本，不改变序号和时间轴
2. 原字幕可能一条对应多条译文，也可能多条对应一条译文，请按时间轴内容综合判断
3. 译文已经正确时原样返回
4. 保持自然、口语化、适合影视字幕
5. 句末通常不加句号，保留必要的问号、叹号、省略号和破折号
6. 输出合法 JSON：{"corrections": [{"id": 0, "text": "修正后译文", "issue": "简短问题说明或 OK"}]}"""

    user_prompt = f"""请质检并修正以下字幕条目：
{json.dumps(items, ensure_ascii=False)}

只输出 JSON，不要解释："""
    return system_prompt, user_prompt


async def _check_quality_chunk(
    client: Any,
    model: str,
    sem: asyncio.Semaphore,
    chunk: List[SrtEntry],
    original_entries: Sequence[SrtEntry],
    config: TranslatorConfig,
) -> Dict[int, QualityIssue]:
    """Run one quality-check chunk."""
    async with sem:
        items: List[Dict[str, Any]] = []
        for local_id, translated in enumerate(chunk):
            aligned = align_originals_by_timeline(original_entries, translated)
            original_text = " ".join(entry.text for entry in aligned)
            original_times = [entry.timecode for entry in aligned]
            items.append(
                {
                    "id": local_id,
                    "timecode": translated.timecode,
                    "original_timecodes": original_times,
                    "original": original_text,
                    "translated": translated.text,
                }
            )

        system_prompt, user_prompt = _build_quality_prompt(items)
        json_str = await call_llm_async(
            client,
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_retries=3,
            json_mode=True,
            max_tokens=config.max_output_tokens,
        )
        return _parse_quality_response(json_str, len(chunk))


async def run_quality_check(
    original_entries: List[SrtEntry],
    translated_entries: List[SrtEntry],
    client: Any,
    config: TranslatorConfig,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> List[SrtEntry]:
    """Return corrected subtitles using original subtitles as timeline-aligned context."""
    if not original_entries:
        raise ValueError("原字幕文件为空或格式不正确。")
    if not translated_entries:
        raise ValueError("译文字幕文件为空或格式不正确。")

    chunk_size = max(1, min(config.chunk_size, 20))
    chunks = [
        translated_entries[i : i + chunk_size]
        for i in range(0, len(translated_entries), chunk_size)
    ]
    sem = asyncio.Semaphore(config.concurrency)

    async def _safe_check_chunk(idx: int, chunk: List[SrtEntry]) -> tuple[int, Dict[int, QualityIssue]]:
        try:
            issues = await _check_quality_chunk(
                client,
                config.model_name,
                sem,
                chunk,
                original_entries,
                config,
            )
            return idx, issues
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Quality check chunk %s failed: %s", idx, exc)
            return idx, {}

    tasks = [_safe_check_chunk(idx, chunk) for idx, chunk in enumerate(chunks)]

    corrected_texts: Dict[int, str] = {}
    total = len(chunks)
    completed = 0

    for coro in asyncio.as_completed(tasks):
        chunk_idx, issues = await coro
        start = chunk_idx * chunk_size
        for local_id, issue in issues.items():
            corrected_texts[start + local_id] = issue.corrected_text

        completed += 1
        if on_progress:
            await on_progress(chunk_idx, int((completed / total) * 100))

    return [
        entry.copy(text=corrected_texts.get(idx, entry.text))
        for idx, entry in enumerate(translated_entries)
    ]
