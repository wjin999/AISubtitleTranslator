"""Data models for SRT subtitle entries."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SrtEntry:
    """Represents a single subtitle entry in SRT format."""
    
    index: int
    start: str
    end: str
    text: str
    
    # 缓存时间戳解析结果
    _start_cache: Optional[float] = field(default=None, repr=False, compare=False)
    _end_cache: Optional[float] = field(default=None, repr=False, compare=False)

    @property
    def timecode(self) -> str:
        """Return the timecode line in SRT format."""
        return f"{self.start} --> {self.end}"
    
    @property
    def start_seconds(self) -> float:
        """Convert start timecode to seconds (cached)."""
        if self._start_cache is None:
            self._start_cache = self._time_str_to_seconds(self.start)
        return self._start_cache

    @property
    def end_seconds(self) -> float:
        """Convert end timecode to seconds (cached)."""
        if self._end_cache is None:
            self._end_cache = self._time_str_to_seconds(self.end)
        return self._end_cache

    @staticmethod
    def _time_str_to_seconds(t_str: str) -> float:
        """Convert SRT timecode string to seconds."""
        try:
            h, m, s_full = t_str.split(':')
            s, ms = s_full.split(',')
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        except (ValueError, AttributeError):
            return 0.0

    def to_srt(self, new_idx: int | None = None) -> str:
        """Convert entry to SRT format string."""
        idx = new_idx if new_idx is not None else self.index
        return f"{idx}\n{self.timecode}\n{self.text}\n\n"
    
    def copy(self, **changes) -> "SrtEntry":
        """Create a copy with optional field changes."""
        return SrtEntry(
            index=changes.get('index', self.index),
            start=changes.get('start', self.start),
            end=changes.get('end', self.end),
            text=changes.get('text', self.text),
        )
