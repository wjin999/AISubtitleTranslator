"""SRT file parsing and saving utilities."""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import List, Sequence, Optional

from .models import SrtEntry

logger = logging.getLogger(__name__)


def parse_srt(content: str) -> List[SrtEntry]:
    """
    Parse SRT file content into list of SrtEntry objects.
    
    Args:
        content: Raw SRT file content as string
        
    Returns:
        List of parsed SrtEntry objects
    """
    if not content or not content.strip():
        return []
    
    # 预处理：标准化换行符，确保末尾有空行
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    content = content.strip() + '\n\n'
    
    # 修复后的正则：支持文件末尾匹配
    pattern = (
        r"(\d+)\s*\n"                                              # 序号
        r"\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"                   # 开始时间
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*\n"                          # 结束时间
        r"([\s\S]*?)(?=\n\s*\n\d+\s*\n|\n\s*\n\s*$|\s*$)"          # 文本内容
    )
    
    entries: List[SrtEntry] = []
    
    for match in re.finditer(pattern, content):
        idx, start, end, text = match.groups()
        # 清理多行文本为单行
        clean_text = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
        if clean_text:  # 只添加有内容的条目
            entries.append(SrtEntry(int(idx), start, end, clean_text))
    
    if not entries:
        logger.warning("No valid SRT entries found in content")
    
    return entries


def validate_srt_file(path: Path) -> Optional[str]:
    """
    Validate SRT file before processing.
    
    Args:
        path: Path to SRT file
        
    Returns:
        Error message if invalid, None if valid
    """
    if not path.exists():
        return f"File not found: {path}"
    
    if not path.is_file():
        return f"Not a file: {path}"
    
    suffix = path.suffix.lower()
    if suffix != '.srt':
        return f"Invalid file extension: {suffix} (expected .srt)"
    
    # 检查文件大小
    size = path.stat().st_size
    if size == 0:
        return "File is empty"
    if size > 50 * 1024 * 1024:  # 50MB
        return f"File too large: {size / 1024 / 1024:.1f}MB (max 50MB)"
    
    return None


def save_srt(entries: Sequence[SrtEntry], path: Path) -> None:
    """
    Save SrtEntry list to SRT file.
    
    Args:
        entries: Sequence of SrtEntry objects to save
        path: Output file path
    """
    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8") as f:
        for new_idx, e in enumerate(entries, 1):
            f.write(e.to_srt(new_idx))
    
    logger.info(f"Saved {len(entries)} entries to {path}")
