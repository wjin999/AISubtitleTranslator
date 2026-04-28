"""Command-line interface for SRT Translator."""

from __future__ import annotations

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

from .config import TranslatorConfig, DEFAULT_GLOSSARY_FILENAME
from .models import SrtEntry
from .parser import parse_srt, save_srt, validate_srt_file
from .merger import init_spacy_model, merge_entries_batch
from .glossary import load_glossary, Glossary
from .llm_client import create_client
from .progress import (
    TranslationProgress,
    get_progress_file,
    save_progress,
    load_progress,
    delete_progress,
)
from .pipeline import TranslationPipeline


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
    parser.add_argument("--model", dest="model_name", default="deepseek-v4-flash")
    parser.add_argument("--summary-model", dest="summary_model_name", default="deepseek-v4-pro")
    
    # Performance
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--chunk-size", dest="chunk_size_for_translation", type=int, default=10)
    
    # Progress
    parser.add_argument("--resume", action="store_true", help="Resume from saved progress")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress saving")
    
    # Misc
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    return parser.parse_args()


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
    
    # 初始化进度
    total_chunks = (len(merged_entries) + config.chunk_size - 1) // config.chunk_size
    if not progress:
        progress = TranslationProgress.create(str(in_path), total_chunks)
    
    # 创建 tqdm 进度条
    pbar = tqdm(total=total_chunks, desc="Translating", unit="chunk")
    
    # 进度回调：更新 tqdm 进度条 + 保存到磁盘
    def _update_progress(chunk_idx: int, pct: int):
        pbar.update(1)
        if progress_path:
            save_progress(progress, progress_path)
    
    # 创建管道并执行翻译
    pipeline = TranslationPipeline(config)
    translations = await pipeline.run(
        entries=merged_entries,
        glossary=glossary,
        client=client,
        progress=progress,
        on_progress=_update_progress,
    )
    
    pbar.close()
    
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
