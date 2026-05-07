"""Integration tests for TranslationPipeline."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from srt_translator.config import TranslatorConfig
from srt_translator.models import SrtEntry
from srt_translator.glossary import Glossary
from srt_translator.pipeline import TranslationPipeline


@pytest.fixture
def two_entries():
    """Create 2 sample SRT entries (fits in 1 chunk with chunk_size=2)."""
    return [
        SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello world"),
        SrtEntry(2, "00:00:04,000", "00:00:06,500", "How are you?"),
    ]


@pytest.fixture
def mock_config():
    """Create a minimal config for testing."""
    return TranslatorConfig(
        api_key="test-key",
        base_url="https://test.api.com",
        model_name="test-model",
        summary_model_name="test-model",
        chunk_size=2,
        concurrency=2,
    )


@pytest.fixture
def mock_client():
    """Create a mock AsyncOpenAI client."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"translations": [{"id": 0, "text": "你好世界"}, {"id": 1, "text": "你好吗？"}]}'
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    return client


CHUNK_RESPONSE = '{"translations": [{"id": 0, "text": "你好世界"}, {"id": 1, "text": "你好吗？"}]}'


@pytest.mark.asyncio
async def test_pipeline_run_basic(two_entries, mock_config, mock_client):
    """Test basic pipeline run with mock LLM."""
    pipeline = TranslationPipeline(mock_config)
    glossary = Glossary()

    with patch("srt_translator.translator.call_llm_async", new_callable=AsyncMock) as mock_call:
        # 2 calls: 1 for summary, 1 for the single chunk
        mock_call.side_effect = [
            "这是一个测试摘要。",  # summary
            CHUNK_RESPONSE,       # chunk 0
        ]

        results = await pipeline.run(
            entries=two_entries,
            glossary=glossary,
            client=mock_client,
        )

    assert len(results) == 2
    assert results[0] == "你好世界"
    assert results[1] == "你好吗？"


@pytest.mark.asyncio
async def test_pipeline_run_empty_entries(mock_config, mock_client):
    """Test pipeline with empty entries list."""
    pipeline = TranslationPipeline(mock_config)
    glossary = Glossary()

    results = await pipeline.run(
        entries=[],
        glossary=glossary,
        client=mock_client,
    )

    assert results == {}


@pytest.mark.asyncio
async def test_pipeline_run_with_progress_callback(two_entries, mock_config, mock_client):
    """Test pipeline with progress callback."""
    pipeline = TranslationPipeline(mock_config)
    glossary = Glossary()
    progress_log = []

    async def on_progress(chunk_idx: int, pct: int):
        progress_log.append((chunk_idx, pct))

    with patch("srt_translator.translator.call_llm_async", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [
            "这是一个测试摘要。",
            CHUNK_RESPONSE,
        ]

        await pipeline.run(
            entries=two_entries,
            glossary=glossary,
            client=mock_client,
            on_progress=on_progress,
        )

    assert len(progress_log) >= 1
    for chunk_idx, pct in progress_log:
        assert isinstance(chunk_idx, int)
        assert isinstance(pct, int)
        assert 0 <= pct <= 100


@pytest.mark.asyncio
async def test_pipeline_run_with_glossary(two_entries, mock_config, mock_client):
    """Test pipeline with glossary terms."""
    pipeline = TranslationPipeline(mock_config)
    glossary = Glossary()
    glossary.add("Hello world", "你好世界")

    with patch("srt_translator.translator.call_llm_async", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [
            "这是一个测试摘要。",
            CHUNK_RESPONSE,
        ]

        results = await pipeline.run(
            entries=two_entries,
            glossary=glossary,
            client=mock_client,
        )

    assert len(results) == 2
    assert results[0] == "你好世界"


@pytest.mark.asyncio
async def test_pipeline_run_with_custom_prompts(two_entries, mock_config, mock_client):
    """Test pipeline with custom summary and translation prompts."""
    pipeline = TranslationPipeline(mock_config)
    glossary = Glossary()

    with patch("srt_translator.translator.call_llm_async", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [
            "Custom summary result.",
            CHUNK_RESPONSE,
        ]

        results = await pipeline.run(
            entries=two_entries,
            glossary=glossary,
            client=mock_client,
            summary_prompt="Custom summary prompt for testing.",
            translation_prompt="Custom translation prompt for testing.",
        )

    assert len(results) == 2
