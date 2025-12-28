"""
SRT Translator - Async LLM-powered subtitle translator with smart merging.

Features:
- Intelligent subtitle merging using spaCy NLP
- Context-aware translation with LLM
- Glossary support for consistent terminology
- Async processing for high performance
- Progress saving and resume support
"""

__version__ = "1.1.0"
__author__ = "wjin999"

from .models import SrtEntry
from .parser import parse_srt, save_srt, validate_srt_file
from .merger import merge_entries, merge_entries_batch, init_spacy_model
from .translator import translate_chunk_task, generate_context_summary, TranslationResult
from .glossary import load_glossary, Glossary, find_matching_terms
from .text_utils import clean_translated_text, validate_translation
from .config import TranslatorConfig
from .progress import TranslationProgress, save_progress, load_progress

__all__ = [
    # Models
    "SrtEntry",
    "TranslationResult",
    "TranslationProgress",
    "TranslatorConfig",
    "Glossary",
    # Parsing
    "parse_srt",
    "save_srt",
    "validate_srt_file",
    # Merging
    "merge_entries",
    "merge_entries_batch",
    "init_spacy_model",
    # Translation
    "translate_chunk_task",
    "generate_context_summary",
    # Glossary
    "load_glossary",
    "find_matching_terms",
    # Utils
    "clean_translated_text",
    "validate_translation",
    # Progress
    "save_progress",
    "load_progress",
]
