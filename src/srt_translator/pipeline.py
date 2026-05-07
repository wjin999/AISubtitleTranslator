"""Unified translation pipeline shared by CLI and API Server."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
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
from .progress import TranslationProgress, save_progress_incremental

logger = logging.getLogger(__name__)


class TranslationPipeline:
    """Unified translation pipeline shared by CLI and API Server."""

    def __init__(self, config: TranslatorConfig):
        self.config = config
        self._progress_path: Optional[Path] = None
        self._translation_memory: Dict[str, str] = {}
        self._file_lock: asyncio.Lock = asyncio.Lock()

    async def run(
        self,
        entries: List[SrtEntry],
        glossary: Glossary,
        client: AsyncOpenAI,
        progress: TranslationProgress | None = None,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
        existing_summary: str | None = None,
        summary_prompt: str | None = None,
        translation_prompt: str | None = None,
        progress_path: Path | None = None,
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
            translation_prompt: Optional custom prompt for translation.
            progress_path: Optional path for incremental progress file saving.

        Returns:
            Dict mapping global_index -> translated_text.
        """
        config = self.config
        self._progress_path = progress_path

        all_texts = [e.text for e in entries]
        chunk_size = config.chunk_size
        chunks = [
            all_texts[i : i + chunk_size]
            for i in range(0, len(all_texts), chunk_size)
        ]
        total_chunks = len(chunks)

        # 3. Generate context summary - 均匀5段采样
        summary = existing_summary or ""
        if not summary:
            num_segments = 5
            total_entries = len(all_texts)
            if total_entries <= num_segments * 2:
                truncated = "\n".join(all_texts)
            else:
                sampled_texts = []
                for i in range(num_segments):
                    idx = int(i * total_entries / num_segments)
                    sampled_texts.append(all_texts[idx])
                truncated = "\n".join(sampled_texts)
            summary = await generate_context_summary(
                truncated, client, config.summary_model_name,
                custom_prompt=summary_prompt,
            )

        # 4. Determine which chunks need translation
        if progress:
            pending = progress.get_pending_chunks()
        else:
            pending = list(range(total_chunks))

        results_dict: Dict[int, str] = {}
        if progress:
            results_dict.update(progress.translations)

        if not pending:
            return results_dict

        sem = asyncio.Semaphore(config.concurrency)
        ctx_window = config.context_window

        # 7. Execute tasks with progress tracking
        coro_to_idx = {
            self._safe_translate_chunk(
                i, chunks, all_texts, summary, glossary, client,
                progress, sem, ctx_window, translation_prompt,
            ): i for i in pending
        }
        total_pending = len(pending)
        completed = 0

        for coro in asyncio.as_completed(coro_to_idx):
            chunk_idx, chunk_translations = await coro
            # Accumulate results (with string keys for JSON compatibility)
            for idx, text in chunk_translations.items():
                results_dict[idx] = text
            completed += 1
            pct = int((completed / total_pending) * 100)
            if on_progress:
                await on_progress(chunk_idx, pct)

        return results_dict

    async def _safe_translate_chunk(
        self,
        chunk_idx: int,
        chunks: List[List[str]],
        all_texts: List[str],
        summary: str,
        glossary: Glossary,
        client: AsyncOpenAI,
        progress: TranslationProgress | None,
        sem: asyncio.Semaphore,
        ctx_window: int,
        translation_prompt: str | None,
    ) -> tuple[int, Dict[int, str]]:
        """Wrapper that prevents a single chunk failure from crashing the pipeline."""
        try:
            return await self._translate_one_chunk(
                chunk_idx, chunks, all_texts, summary, glossary, client,
                progress, sem, ctx_window, translation_prompt,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to translate chunk {chunk_idx}: {e}")
            return chunk_idx, {}

    async def _translate_one_chunk(
        self,
        chunk_idx: int,
        chunks: List[List[str]],
        all_texts: List[str],
        summary: str,
        glossary: Glossary,
        client: AsyncOpenAI,
        progress: TranslationProgress | None,
        sem: asyncio.Semaphore,
        ctx_window: int,
        translation_prompt: str | None,
    ) -> tuple[int, Dict[int, str]]:
        """Translate a single chunk and return (chunk_idx, {global_index: text})."""
        chunk = chunks[chunk_idx]
        chunk_size = self.config.chunk_size
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + len(chunk)

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
            model=self.config.model_name,
            sem=sem,
            custom_translation_prompt=translation_prompt,
            translation_memory=self._translation_memory,
        )

        chunk_translations: Dict[int, str] = {}
        for r in results:
            if r.success:
                cleaned = clean_translated_text(r.translated)
                chunk_translations[r.index] = cleaned
                self._translation_memory[r.original] = cleaned
            else:
                chunk_translations[r.index] = r.original

        # 限制翻译记忆大小
        if len(self._translation_memory) > 200:
            items = list(self._translation_memory.items())
            self._translation_memory = dict(items[-100:])

        # 增量保存进度
        if progress:
            new_results = progress.mark_completed(chunk_idx, chunk_translations)
            if self._progress_path:
                async with self._file_lock:
                    save_progress_incremental(progress, self._progress_path, new_results, chunk_idx)

        return chunk_idx, chunk_translations
