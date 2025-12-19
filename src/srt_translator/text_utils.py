"""Text processing utilities."""

from __future__ import annotations

import re
import string

# Punctuation characters to normalize
_PUNCT_ASCII = string.punctuation
_PUNCT_CHINESE = "、。，？！：；（）【】「」『』«»‹›〈〉《》""…—–―─～·"
ALL_PUNCTUATION = "".join(sorted(set(_PUNCT_ASCII + _PUNCT_CHINESE)))


def clean_translated_text(
    text: str, 
    remove_punctuation: bool = True
) -> str:
    """
    Clean and normalize translated subtitle text.
    
    Args:
        text: Raw translated text
        remove_punctuation: Whether to replace punctuation with spaces
        
    Returns:
        Cleaned text
    """
    # Remove leading list markers (e.g., "1.", "2)", "-", "*")
    text = re.sub(r'^\s*(\d+[\.:\)]\s*|[-*\u2022]\s*)', '', text.strip())
    
    # Replace punctuation with spaces if requested
    if remove_punctuation:
        escaped = re.escape(ALL_PUNCTUATION)
        text = re.sub(f"[{escaped}]+", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
