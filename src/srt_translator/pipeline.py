"""Unified translation pipeline shared by CLI and API Server."""

from __future__ import annotations

import asyncio
import inspect
from typing import Callable, Dict, List, Awaitable, Optional

from openai import AsyncOpenAI

from .config import TranslatorConfig
from .models import SrtEntry
from .glossary import Glossary
from .translator import (
    translate_chunk_task,
    generate_context_summary,
)
from .text_utils import clean_translated_text
from .progress import TranslationProgress


class TranslationPipeline:
    """Unified translation pipeline shared by CLI and API Server."""

    def __init__(self, config: TranslatorConfig):
        self.config = config

    async def run(
        self,
        entries: List[SrtEntry],
        glossary: Glossary,
        client: AsyncOpenAI,
        progress: TranslationProgress | None = None,
        on_progress: Callable[[int, int], Awaitable[None] | None] | None = None,
        existing_summary: str | None = None,
        summary_prompt: str | None = None,
        translation_prompt: str | None = None,
    ) -> Dict[int, str]:
        """
        Execute the complete translation pipeline.

        Args:
            entries: Merged SrtEntry list (caller must merge first).
            glossary: Glossary instance for terminology matching.
            client: Pre-configured AsyncOpenAI client.
            progress: Progress tracker for resume support.
            on_progress: Progress callback receiving (chunk_index, progress_pct).
            existing_summary: Pre-existing summary for resume.
            summary_prompt: Optional custom prompt for summary generation.
            translation_prompt: Optional custom prompt for translation (reserved for future use).

        Returns:
            Dict mapping global_index -> translated_text.
        """
        config = self.config

        # 1. Extract all texts in order
        all_texts = [e.text for e in entries]

        # 2. Split into chunks
        chunk_size = config.chunk_size
        chunks = [
            all_texts[i : i + chunk_size]
            for i in range(0, len(all_texts), chunk_size)
        ]
        total_chunks = len(chunks)

        # 3. Generate context summary if not already available
        summary = existing_summary or ""
        if not summary:
            full_text = "\n".join(all_texts)
            summary = await generate_context_summary(
                full_text, client, config.summary_model_name,
                custom_prompt=summary_prompt,
            )

        # 4. Determine which chunks need translation
        if progress:
            pending = progress.get_pending_chunks()
        else:
            pending = list(range(total_chunks))

        # 5. Restore already-completed translations
        results_dict: Dict[int, str] = {}
        if progress:
            results_dict.update(progress.translations)

        if not pending:
            return results_dict

        # 6. Create semaphore for concurrency control
        sem = asyncio.Semaphore(config.concurrency)
        ctx_window = config.context_window

        # 7. Capture translation_prompt for closure use
        _translation_prompt = translation_prompt

        # 8. Build wrapper that returns (chunk_idx, chunk_translations)
        async def _translate_one_chunk(chunk_idx: int) -> tuple[int, Dict[int, str]]:
            """Translate a single chunk and return (chunk_idx, {global_index: text})."""
            chunk = chunks[chunk_idx]
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + len(chunk)

            # Context window
            prev_start = max(0, start_idx - ctx_window)
            context_prev = all_texts[prev_start:start_idx]
            next_end = min(len(all_texts), end_idx + ctx_window)
            context_next = all_texts[end_idx:next_end]

            chunk_data = [
                {"index": start_idx + j, "text": text}
                for j, text in enumerate(chunk)
            ]

            results = await translate_chunk_task(
                client=client,
                chunk_data=chunk_data,
                context_prev=context_prev,
                context_next=context_next,
                global_summary=summary,
                glossary=glossary,
                model=config.model_name,
                sem=sem,
                custom_translation_prompt=_translation_prompt,
            )

            # Process results: clean + validate, fallback to original on failure
            chunk_translations: Dict[int, str] = {}
            for r in results:
                if r.success:
                    cleaned = clean_translated_text(r.translated)
                    results_dict[r.index] = cleaned
                    chunk_translations[r.index] = cleaned
                else:
                    results_dict[r.index] = r.original
                    chunk_translations[r.index] = r.original

            # Update progress
            if progress:
                progress.mark_completed(chunk_idx, chunk_translations)

            return chunk_idx, chunk_translations

        # 8. Execute tasks with progress tracking
        coro_to_idx = {_translate_one_chunk(i): i for i in pending}
        total_pending = len(pending)
        completed = 0

        for coro in asyncio.as_completed(coro_to_idx):
            chunk_idx, _ = await coro
            completed += 1
            pct = int((completed / total_pending) * 100)

            # Notify via callback (supports both sync and async callbacks)
            if on_progress:
                result = on_progress(chunk_idx, pct)
                if inspect.isawaitable(result):
                    await result

        return results_dict
