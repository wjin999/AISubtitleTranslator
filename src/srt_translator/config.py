"""Configuration and constants."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class TranslatorConfig:
    """Configuration for subtitle translator."""
    
    # API settings
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-v4-pro"          # 默认模型，可通过 DEEPSEEK_MODEL 环境变量覆盖
    summary_model_name: str = "deepseek-v4-pro"  # 默认摘要模型，可通过 DEEPSEEK_SUMMARY_MODEL 覆盖
    
    # Processing settings
    concurrency: int = 8
    chunk_size: int = 10
    context_window: int = 7
    
    # Merge settings (spaCy smart merging is always enabled)
    max_chars_per_entry: int = 300
    merge_time_gap: float = 1.5
    
    # Output settings
    output_prefix: str = "translated_"
    
    # Progress settings
    save_progress: bool = True
    progress_file: Optional[Path] = None
    
    # Deprecated model names
    DEPRECATED_MODELS = {"deepseek-reasoner"}
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    @classmethod
    def from_args(cls, args) -> "TranslatorConfig":
        """Create config from argparse namespace.

        Precedence: CLI arg > env var > default ("deepseek-v4-pro").
        """
        api_key = getattr(args, 'api_key', None)
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY")

        model_name = getattr(args, 'model_name', None)
        if model_name is None:
            model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

        summary_model_name = getattr(args, 'summary_model_name', None)
        if summary_model_name is None:
            summary_model_name = os.environ.get("DEEPSEEK_SUMMARY_MODEL", "deepseek-v4-pro")

        return cls(
            api_key=api_key,
            base_url=getattr(args, 'base_url', "https://api.deepseek.com"),
            model_name=model_name,
            summary_model_name=summary_model_name,
            concurrency=getattr(args, 'concurrency', 8),
            chunk_size=getattr(args, 'chunk_size_for_translation', 10),
            context_window=getattr(args, 'context_window', 7),
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
        
        # Check for deprecated models
        if self.model_name in self.DEPRECATED_MODELS:
            logger.warning(
                f"Model '{self.model_name}' is deprecated and will be removed in a future version. "
                f"Please update to a newer model."
            )
        if self.summary_model_name in self.DEPRECATED_MODELS:
            logger.warning(
                f"Summary model '{self.summary_model_name}' is deprecated and will be removed in a future version. "
                f"Please update to a newer model."
            )
        
        return None


# Default glossary filename
DEFAULT_GLOSSARY_FILENAME = "glossary.txt"

# Supported file extensions
SUPPORTED_EXTENSIONS = {".srt"}

# Progress file suffix
PROGRESS_SUFFIX = ".progress.json"
