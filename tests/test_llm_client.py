"""Tests for LLM client request parameter handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from srt_translator.llm_client import call_llm_async


def _mock_client():
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "ok"
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_deepseek_uses_max_tokens():
    client = _mock_client()

    await call_llm_async(
        client,
        "deepseek-v4-pro",
        [{"role": "user", "content": "hello"}],
        max_tokens=123,
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["max_tokens"] == 123
    assert "max_completion_tokens" not in kwargs


@pytest.mark.asyncio
async def test_omits_optional_sampling_and_reasoning_parameters():
    client = _mock_client()

    await call_llm_async(
        client,
        "deepseek-v4-pro",
        [{"role": "user", "content": "hello"}],
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    assert "temperature" not in kwargs
    assert "reasoning_effort" not in kwargs
