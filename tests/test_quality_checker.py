"""Tests for timeline-based subtitle quality checking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from srt_translator.config import TranslatorConfig
from srt_translator.models import SrtEntry
from srt_translator.quality_checker import align_originals_by_timeline, run_quality_check


def test_align_originals_by_timeline_handles_merged_translation():
    originals = [
        SrtEntry(1, "00:00:01,000", "00:00:02,000", "Hello"),
        SrtEntry(2, "00:00:02,000", "00:00:03,000", "world"),
        SrtEntry(3, "00:00:05,000", "00:00:06,000", "Later"),
    ]
    translated = SrtEntry(1, "00:00:01,000", "00:00:03,000", "你好世界")

    aligned = align_originals_by_timeline(originals, translated)

    assert [entry.text for entry in aligned] == ["Hello", "world"]


@pytest.mark.asyncio
async def test_run_quality_check_applies_model_corrections():
    originals = [SrtEntry(1, "00:00:01,000", "00:00:03,000", "Hello world")]
    translated = [SrtEntry(1, "00:00:01,000", "00:00:03,000", "你好")]
    config = TranslatorConfig(
        api_key="test-key",
        model_name="test-model",
        summary_model_name="test-model",
    )

    response = '{"corrections":[{"id":0,"text":"你好世界","issue":"漏译 world"}]}'
    with patch("srt_translator.quality_checker.call_llm_async", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = response
        corrected = await run_quality_check(
            originals,
            translated,
            MagicMock(),
            config,
        )

    assert corrected[0].text == "你好世界"
    assert corrected[0].timecode == translated[0].timecode
    mock_call.assert_awaited_once()
