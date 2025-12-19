"""SRT file parsing and saving utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence

from .models import SrtEntry


def parse_srt(content: str) -> List[SrtEntry]:
    """
    Parse SRT file content into list of SrtEntry objects.
    
    Args:
        content: Raw SRT file content as string
        
    Returns:
        List of parsed SrtEntry objects
    """
    pattern = (
        r"(\d+)\s*\n"
        r"\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
        r"([\s\S]*?)(?=\n\s*\n\s*\d+\s*\n|$)"
    )
    entries: list[SrtEntry] = []
    for idx, start, end, text in re.findall(pattern, content):
        # Clean up multiline text into single line
        clean = " ".join(line.strip() for line in text.strip().splitlines())
        entries.append(SrtEntry(int(idx), start, end, clean))
    return entries


def save_srt(entries: Sequence[SrtEntry], path: Path) -> None:
    """
    Save SrtEntry list to SRT file.
    
    Args:
        entries: Sequence of SrtEntry objects to save
        path: Output file path
    """
    with path.open("w", encoding="utf-8") as f:
        for new_idx, e in enumerate(entries, 1):
            f.write(e.to_srt(new_idx))
