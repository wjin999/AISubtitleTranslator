"""Data models for SRT subtitle entries."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SrtEntry:
    """Represents a single subtitle entry in SRT format."""
    
    index: int
    start: str
    end: str
    text: str

    @property
    def timecode(self) -> str:
        """Return the timecode line in SRT format."""
        return f"{self.start} --> {self.end}"
    
    @property
    def start_seconds(self) -> float:
        """Convert start timecode to seconds."""
        return self._time_str_to_seconds(self.start)

    @property
    def end_seconds(self) -> float:
        """Convert end timecode to seconds."""
        return self._time_str_to_seconds(self.end)

    @staticmethod
    def _time_str_to_seconds(t_str: str) -> float:
        """
        Convert SRT timecode string to seconds.
        
        Args:
            t_str: Timecode in format "HH:MM:SS,mmm"
            
        Returns:
            Total seconds as float
        """
        try:
            h, m, s_full = t_str.split(':')
            s, ms = s_full.split(',')
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        except ValueError:
            return 0.0

    def to_srt(self, new_idx: int | None = None) -> str:
        """
        Convert entry to SRT format string.
        
        Args:
            new_idx: Optional new index to use instead of self.index
            
        Returns:
            Formatted SRT entry string
        """
        idx = new_idx if new_idx is not None else self.index
        return f"{idx}\n{self.timecode}\n{self.text}\n\n"
