"""Configuration and constants."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables once
load_dotenv()


@dataclass
class TranslatorConfig:
    """Configuration for subtitle translator."""
    
    # API settings
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-chat"
    summary_model_name: str = "deepseek-reasoner"
    
    # Processing settings
    concurrency: int = 8
    chunk_size: int = 10
    context_window: int = 3
    
    # Merge settings
    enable_merge: bool = True
    max_chars_per_entry: int = 300
    merge_time_gap: float = 1.5
    
    # Output settings
    output_prefix: str = "translated_"
    
    # Progress settings
    save_progress: bool = True
    progress_file: Optional[Path] = None
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    @classmethod
    def from_args(cls, args) -> "TranslatorConfig":
        """Create config from argparse namespace."""
        # 只从 args 或环境变量读取一次
        api_key = getattr(args, 'api_key', None)
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        
        return cls(
            api_key=api_key,
            base_url=getattr(args, 'base_url', "https://api.deepseek.com"),
            model_name=getattr(args, 'model_name', "deepseek-chat"),
            summary_model_name=getattr(args, 'summary_model_name', "deepseek-reasoner"),
            concurrency=getattr(args, 'concurrency', 8),
            chunk_size=getattr(args, 'chunk_size_for_translation', 10),
            enable_merge=not getattr(args, 'no_merge', False),
            max_chars_per_entry=getattr(args, 'max_chars_per_entry', 300),
            merge_time_gap=getattr(args, 'merge_time_gap', 1.5),
        )
    
    def validate(self) -> Optional[str]:
        """
        Validate configuration.
        
        Returns:
            Error message if invalid, None if valid
        """
        if not self.api_key:
            return "API key is required. Set DEEPSEEK_API_KEY or use --api-key"
        
        if self.concurrency < 1 or self.concurrency > 50:
            return f"Concurrency must be 1-50, got {self.concurrency}"
        
        if self.chunk_size < 1 or self.chunk_size > 50:
            return f"Chunk size must be 1-50, got {self.chunk_size}"
        
        return None


# Default glossary filename
DEFAULT_GLOSSARY_FILENAME = "glossary.txt"

# Supported file extensions
SUPPORTED_EXTENSIONS = {".srt"}

# Progress file suffix
PROGRESS_SUFFIX = ".progress.json"
