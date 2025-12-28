"""Text processing utilities."""

from __future__ import annotations

import re


# 需要清理的标点（不包括中文常用标点）
REMOVABLE_PUNCTUATION = r'[#*\-_=+|\\<>]'

# 中文标点（应该保留）
CHINESE_PUNCTUATION = "，。！？、；：""''（）【】"


def clean_translated_text(text: str) -> str:
    """
    Clean and normalize translated subtitle text.
    
    只清理格式标记，保留正常标点符号。
    
    Args:
        text: Raw translated text
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    # 1. 移除开头的列表标记 (如 "1.", "2)", "-", "*", "•")
    text = re.sub(r'^\s*(\d+[\.:\)]\s*|[-*•]\s*)', '', text)
    
    # 2. 移除 markdown 格式标记
    text = re.sub(r'\*\*|__', '', text)  # 粗体
    text = re.sub(r'\*|_', '', text)      # 斜体
    
    # 3. 移除多余的特殊字符，但保留正常标点
    text = re.sub(REMOVABLE_PUNCTUATION, ' ', text)
    
    # 4. 标准化空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def validate_translation(original: str, translated: str) -> tuple[bool, str]:
    """
    Validate translation quality.
    
    Args:
        original: Original text
        translated: Translated text
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not translated:
        return False, "Empty translation"
    
    if translated.startswith("[Fail]"):
        return False, "Translation failed"
    
    # 检查是否全是乱码/特殊字符
    clean = re.sub(r'[\s\W]', '', translated)
    if not clean:
        return False, "Translation contains only special characters"
    
    # 检查长度异常（译文不应比原文长太多或短太多）
    orig_len = len(original)
    trans_len = len(translated)
    
    if orig_len > 10:  # 只对较长文本检查
        if trans_len < orig_len * 0.1:
            return False, f"Translation too short ({trans_len} vs {orig_len})"
        if trans_len > orig_len * 5:
            return False, f"Translation too long ({trans_len} vs {orig_len})"
    
    # 检查是否只是复制了原文
    if translated.strip().lower() == original.strip().lower():
        return False, "Translation is identical to original"
    
    return True, ""


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
