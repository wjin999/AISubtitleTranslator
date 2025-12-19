"""
SRT Translator - Async LLM-powered subtitle translator with smart merging.

A professional subtitle translation tool featuring:
- Intelligent subtitle merging using spaCy NLP
- Context-aware translation with LLM
- Glossary support for consistent terminology
- Async processing for high performance
"""

__version__ = "1.0.0"
__author__ = "WJ"

from .models import SrtEntry
from .parser import parse_srt, save_srt
from .merger import merge_entries, init_spacy_model
from .translator import translate_chunk_task, generate_context_summary
from .glossary import load_glossary

__all__ = [
    "SrtEntry",
    "parse_srt",
    "save_srt",
    "merge_entries",
    "init_spacy_model",
    "translate_chunk_task",
    "generate_context_summary",
    "load_glossary",
]
