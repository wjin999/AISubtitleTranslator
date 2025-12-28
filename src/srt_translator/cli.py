"""Command-line interface for SRT Translator."""

from __future__ import annotations

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict

from tqdm.asyncio import tqdm_asyncio

from .config import TranslatorConfig, DEFAULT_GLOSSARY_FILENAME
from .models import SrtEntry
from .parser import parse_srt, save_srt, validate_srt_file
from .merger import init_spacy_model, merge_entries_batch
from .glossary import load_glossary, Glossary
from .translator import (
    translate_chunk_task, 
    generate_context_summary,
    TranslationResult,
    extract_translations,
)
from .text_utils import clean_translated_text
from .llm_client import create_client
from .progress import (
    TranslationProgress,
    get_progress_file,
    save_progress,
    load_progress,
    delete_progress,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Async LLM Subtitle Translator with Smart Merging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.srt                     # Basic translation
  %(prog)s video.srt -o output.srt       # Specify output
  %(prog)s video.srt -g glossary.txt     # Use glossary
  %(prog)s video.srt --no-merge          # Disable merging
  %(prog)s video.srt --resume            # Resume interrupted translation
        """
    )
    
    # Positional arguments
    parser.add_argument("input_path", help="Input SRT file path")
    parser.add_argument("output_path", nargs='?', default=None, help="Output SRT file path")
    
    # Glossary
    parser.add_argument("-g", "--glossary", dest="glossary_path", help="Glossary file path")
    
    # Processing options
    parser.add_argument("--no-merge", action="store_true", help="Disable smart merging")
    parser.add_argument("--max-chars", dest="max_chars_per_entry", type=int, default=300)
    parser.add_argument("--merge-gap", dest="merge_time_gap", type=float, default=1.5)
    
    # API options
    parser.add_argument("--api-key", help="API key (or set DEEPSEEK_API_KEY)")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", dest="model_name", default="deepseek-chat")
    parser.add_argument("--summary-model", dest="summary_model_name", default="deepseek-reasoner")
    
    # Performance
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--chunk-size", dest="chunk_size_for_translation", type=int, default=10)
    
    # Progress
    parser.add_argument("--resume", action="store_true", help="Resume from saved progress")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress saving")
    
    # Misc
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    return parser.parse_args()


async def run_translation(
    client,
    chunks: List[List[str]],
    all_texts: List[str],
    summary: str,
    glossary: Glossary,
    config: TranslatorConfig,
    progress: TranslationProgress | None,
    progress_path: Path | None,
) -> Dict[int, str]:
    """
    Execute translation with progress tracking.
    
    Returns:
        Dict mapping global index to translated text
    """
    logger = logging.getLogger(__name__)
    sem = asyncio.Semaphore(config.concurrency)
    chunk_size = config.chunk_size
    ctx_window = config.context_window
    
    # 确定需要翻译的 chunks
    if progress:
        pending = progress.get_pending_chunks()
        logger.info(f"Resuming: {len(progress.completed_chunks)}/{progress.total_chunks} chunks done")
    else:
        pending = list(range(len(chunks)))
    
    results_dict: Dict[int, str] = {}
    
    # 从进度恢复已完成的翻译
    if progress:
        results_dict.update(progress.translations)
    
    # 创建待翻译任务
    tasks = []
    task_chunk_indices = []
    
    for chunk_idx in pending:
        chunk = chunks[chunk_idx]
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + len(chunk)
        
        # 上下文窗口
        prev_start = max(0, start_idx - ctx_window)
        context_prev = all_texts[prev_start:start_idx]
        next_end = min(len(all_texts), end_idx + ctx_window)
        context_next = all_texts[end_idx:next_end]
        
        chunk_data = [
            {"index": start_idx + j, "text": text}
            for j, text in enumerate(chunk)
        ]
        
        task = translate_chunk_task(
            client=client,
            chunk_data=chunk_data,
            context_prev=context_prev,
            context_next=context_next,
            global_summary=summary,
            glossary=glossary,
            model=config.model_name,
            sem=sem,
        )
        tasks.append(task)
        task_chunk_indices.append(chunk_idx)
    
    if not tasks:
        logger.info("No chunks to translate")
        return results_dict
    
    logger.info(f"Translating {len(tasks)} chunks...")
    
    # 执行翻译
    results_list = await tqdm_asyncio.gather(*tasks, desc="Translating")
    
    # 处理结果
    for chunk_idx, chunk_results in zip(task_chunk_indices, results_list):
        chunk_translations = {}
        
        for r in chunk_results:
            if r.success:
                cleaned = clean_translated_text(r.translated)
                results_dict[r.index] = cleaned
                chunk_translations[r.index] = cleaned
            else:
                # 失败时保留原文
                results_dict[r.index] = r.original
                chunk_translations[r.index] = r.original
                logger.warning(f"Translation failed for #{r.index}: {r.error}")
        
        # 更新进度
        if progress and progress_path:
            progress.mark_completed(chunk_idx, chunk_translations)
            save_progress(progress, progress_path)
    
    return results_dict


async def main_async(args: argparse.Namespace) -> int:
    """Main async workflow."""
    logger = logging.getLogger(__name__)
    config = TranslatorConfig.from_args(args)
    
    # 验证配置
    error = config.validate()
    if error:
        logger.error(error)
        return 1
    
    # 验证输入文件
    in_path = Path(args.input_path).expanduser().resolve()
    error = validate_srt_file(in_path)
    if error:
        logger.error(error)
        return 1
    
    # 读取并解析 SRT
    logger.info(f"Reading: {in_path}")
    content = in_path.read_text(encoding="utf-8-sig")
    entries = parse_srt(content)
    
    if not entries:
        logger.error("No valid subtitle entries found")
        return 1
    
    logger.info(f"Parsed {len(entries)} subtitle entries")
    
    # 加载术语表
    glossary = Glossary()
    if args.glossary_path:
        g_path = Path(args.glossary_path).expanduser().resolve()
        glossary = load_glossary(g_path)
    elif Path(DEFAULT_GLOSSARY_FILENAME).exists():
        logger.info(f"Auto-detected '{DEFAULT_GLOSSARY_FILENAME}'")
        glossary = load_glossary(Path(DEFAULT_GLOSSARY_FILENAME))
    
    # 智能合并
    if config.enable_merge:
        init_spacy_model()
        merged_entries = merge_entries_batch(
            entries, 
            config.max_chars_per_entry, 
            config.merge_time_gap
        )
    else:
        logger.info("Merging disabled")
        merged_entries = [e.copy() for e in entries]
    
    all_texts = [e.text for e in merged_entries]
    
    # 进度管理
    progress_path = get_progress_file(in_path) if not args.no_progress else None
    progress = None
    
    if args.resume and progress_path:
        progress = load_progress(progress_path)
        if progress:
            logger.info(f"Loaded progress: {progress.completion_rate:.0%} complete")
        else:
            logger.info("No previous progress found, starting fresh")
    
    # 创建 API 客户端
    client = create_client(config.api_key, config.base_url)
    
    # 准备 chunks
    chunk_size = config.chunk_size
    chunks = [
        all_texts[i:i + chunk_size] 
        for i in range(0, len(all_texts), chunk_size)
    ]
    
    # 初始化进度
    if not progress:
        progress = TranslationProgress.create(str(in_path), len(chunks))
    
    # 生成摘要（可以和第一批翻译并行，但这里简化处理）
    summary = ""
    if not progress.completed_chunks:
        full_text = "\n".join(all_texts)
        summary = await generate_context_summary(
            full_text, client, config.summary_model_name
        )
    
    # 执行翻译
    translations = await run_translation(
        client, chunks, all_texts, summary, glossary,
        config, progress, progress_path
    )
    
    # 构建输出
    final_entries: List[SrtEntry] = []
    for i, entry in enumerate(merged_entries):
        translated_text = translations.get(i, entry.text)
        final_entries.append(entry.copy(text=translated_text))
    
    # 保存结果
    if args.output_path:
        out_path = Path(args.output_path)
    else:
        out_path = in_path.with_name(f"{config.output_prefix}{in_path.name}")
    
    save_srt(final_entries, out_path)
    
    # 清理进度文件
    if progress_path and progress_path.exists():
        delete_progress(progress_path)
    
    # 统计
    success_count = sum(1 for r in translations.values() if r)
    logger.info(f"Done! {success_count}/{len(merged_entries)} translated. Saved to {out_path}")
    
    return 0


def main() -> None:
    """CLI entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    try:
        exit_code = asyncio.run(main_async(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress saved.")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
