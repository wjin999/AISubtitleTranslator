"""Configuration and constants."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class TranslatorConfig:
    """Configuration for subtitle translator."""
    
    # API settings
    api_key: Optional[str] = field(default=None)
    base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-chat"
    summary_model_name: str = "deepseek-reasoner"
    
    # Processing settings
    concurrency: int = 8
    chunk_size: int = 10
    context_window: int = 2
    
    # Merge settings
    enable_merge: bool = True
    max_chars_per_entry: int = 300
    merge_time_gap: float = 1.5
    
    # Output settings
    output_prefix: str = "translated_"
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    @classmethod
    def from_args(cls, args) -> "TranslatorConfig":
        """Create config from argparse namespace."""
        return cls(
            api_key=args.api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=args.base_url,
            model_name=args.model_name,
            summary_model_name=args.summary_model_name,
            concurrency=args.concurrency,
            chunk_size=args.chunk_size_for_translation,
            enable_merge=not args.no_merge,
            max_chars_per_entry=args.max_chars_per_entry,
            merge_time_gap=args.merge_time_gap,
        )


# Default glossary filename
DEFAULT_GLOSSARY_FILENAME = "glossary.txt"

# Supported file extensions
SUPPORTED_EXTENSIONS = {".srt"}
