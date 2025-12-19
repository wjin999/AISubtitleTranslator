"""Command-line interface for SRT Translator."""

from __future__ import annotations

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from tqdm.asyncio import tqdm_asyncio

from .config import TranslatorConfig, DEFAULT_GLOSSARY_FILENAME
from .models import SrtEntry
from .parser import parse_srt, save_srt
from .merger import init_spacy_model, merge_entries
from .glossary import load_glossary
from .translator import translate_chunk_task, generate_context_summary
from .text_utils import clean_translated_text
from .llm_client import create_client


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s: %(message)s"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Async LLM Subtitle Translator with Smart Merging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.srt                          # Basic translation
  %(prog)s video.srt -o translated.srt        # Specify output
  %(prog)s video.srt --glossary terms.txt     # Use custom glossary
  %(prog)s video.srt --no-merge               # Disable smart merging
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "input_path",
        help="Path to input SRT file"
    )
    parser.add_argument(
        "output_path",
        nargs='?',
        default=None,
        help="Path to output SRT file (default: translated_<input>)"
    )
    
    # Glossary options
    parser.add_argument(
        "--glossary", "-g",
        dest="glossary_path",
        help="Path to glossary file (format: Term=Translation)"
    )
    
    # Processing options
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable smart merging (translate line-by-line)"
    )
    parser.add_argument(
        "--max-chars",
        dest="max_chars_per_entry",
        type=int,
        default=300,
        help="Maximum characters per merged entry (default: 300)"
    )
    parser.add_argument(
        "--merge-gap",
        dest="merge_time_gap",
        type=float,
        default=1.5,
        help="Maximum time gap (seconds) for merging (default: 1.5)"
    )
    
    # API options
    parser.add_argument(
        "--api-key",
        help="API key (default: DEEPSEEK_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="API base URL (default: https://api.deepseek.com)"
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        default="deepseek-chat",
        help="Translation model (default: deepseek-chat)"
    )
    parser.add_argument(
        "--summary-model",
        dest="summary_model_name",
        default="deepseek-reasoner",
        help="Summary model (default: deepseek-reasoner)"
    )
    
    # Performance options
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent API requests (default: 8)"
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size_for_translation",
        type=int,
        default=10,
        help="Lines per translation chunk (default: 10)"
    )
    
    # Misc
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    """
    Main async workflow.
    
    Returns:
        Exit code (0 for success)
    """
    logger = logging.getLogger(__name__)
    config = TranslatorConfig.from_args(args)
    
    # Validate API key
    if not config.api_key:
        logger.error(
            "Missing API Key. Set DEEPSEEK_API_KEY env var or use --api-key."
        )
        return 1
    
    # Validate input file
    in_path = Path(args.input_path).expanduser().resolve()
    if not in_path.exists():
        logger.error(f"File not found: {in_path}")
        return 1
    
    logger.info(f"Reading: {in_path}")
    original_content = in_path.read_text(encoding="utf-8-sig")
    entries = parse_srt(original_content)
    logger.info(f"Parsed {len(entries)} subtitle entries")
    
    # Load glossary (priority: CLI arg > auto-detect > none)
    glossary = {}
    if args.glossary_path:
        g_path = Path(args.glossary_path).expanduser().resolve()
        glossary = load_glossary(g_path)
    elif Path(DEFAULT_GLOSSARY_FILENAME).exists():
        logger.info(f"Auto-detected '{DEFAULT_GLOSSARY_FILENAME}' in current directory")
        glossary = load_glossary(Path(DEFAULT_GLOSSARY_FILENAME))
    else:
        logger.info("No glossary loaded")
    
    # Smart merging
    if config.enable_merge:
        init_spacy_model()
        logger.info("Merging entries with spaCy...")
        merged_entries = merge_entries(
            entries, 
            config.max_chars_per_entry, 
            config.merge_time_gap
        )
        logger.info(f"Merged {len(entries)} -> {len(merged_entries)} entries")
    else:
        logger.info("Merging disabled")
        merged_entries = entries
    
    all_texts = [e.text for e in merged_entries]
    
    # Create API client
    client = create_client(config.api_key, config.base_url)
    
    # Generate context summary
    full_text = "\n".join(all_texts)
    summary = await generate_context_summary(
        full_text, 
        client, 
        config.summary_model_name
    )
    
    # Prepare translation chunks
    chunk_size = config.chunk_size
    chunks = [
        all_texts[i:i + chunk_size] 
        for i in range(0, len(all_texts), chunk_size)
    ]
    
    sem = asyncio.Semaphore(config.concurrency)
    tasks = []
    ctx_window = config.context_window
    
    logger.info(f"Starting translation ({len(chunks)} chunks)...")
    
    for i, chunk in enumerate(chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + len(chunk)
        
        # Get context windows
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
            sem=sem
        )
        tasks.append(task)
    
    # Execute all translation tasks
    results_list = await tqdm_asyncio.gather(*tasks, desc="Translating")
    
    # Flatten results
    final_translated = []
    for res in results_list:
        final_translated.extend(res)
    
    # Handle length mismatch
    if len(final_translated) != len(merged_entries):
        logger.warning(
            f"Length mismatch: {len(final_translated)} translations "
            f"for {len(merged_entries)} entries"
        )
    
    # Build output entries
    min_len = min(len(final_translated), len(merged_entries))
    final_entries: List[SrtEntry] = []
    
    for i in range(min_len):
        clean_text = clean_translated_text(final_translated[i])
        orig = merged_entries[i]
        final_entries.append(SrtEntry(orig.index, orig.start, orig.end, clean_text))
    
    # Determine output path
    if args.output_path:
        out_path = Path(args.output_path)
    else:
        out_path = in_path.with_name(f"{config.output_prefix}{in_path.name}")
    
    save_srt(final_entries, out_path)
    logger.info(f"Done! Saved to {out_path}")
    
    return 0


def main() -> None:
    """CLI entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    try:
        exit_code = asyncio.run(main_async(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
