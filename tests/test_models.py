"""Tests for SrtEntry model."""

import pytest
from srt_translator.models import SrtEntry


class TestSrtEntry:
    """Test cases for SrtEntry dataclass."""
    
    def test_creation(self):
        """Test basic creation."""
        entry = SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello world")
        assert entry.index == 1
        assert entry.start == "00:00:01,000"
        assert entry.end == "00:00:03,500"
        assert entry.text == "Hello world"
    
    def test_timecode_property(self):
        """Test timecode formatting."""
        entry = SrtEntry(1, "00:00:01,000", "00:00:03,500", "Test")
        assert entry.timecode == "00:00:01,000 --> 00:00:03,500"
    
    def test_start_seconds(self):
        """Test conversion to seconds."""
        entry = SrtEntry(1, "01:30:45,500", "01:30:50,000", "Test")
        # 1*3600 + 30*60 + 45 + 0.5 = 5445.5
        assert entry.start_seconds == 5445.5
    
    def test_end_seconds(self):
        """Test end time conversion."""
        entry = SrtEntry(1, "00:00:00,000", "00:01:30,250", "Test")
        # 90 + 0.25 = 90.25
        assert entry.end_seconds == 90.25
    
    def test_invalid_timecode(self):
        """Test handling of invalid timecode."""
        entry = SrtEntry(1, "invalid", "00:00:01,000", "Test")
        assert entry.start_seconds == 0.0
    
    def test_to_srt(self):
        """Test SRT format output."""
        entry = SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello world")
        expected = "1\n00:00:01,000 --> 00:00:03,500\nHello world\n\n"
        assert entry.to_srt() == expected
    
    def test_to_srt_with_new_index(self):
        """Test SRT output with custom index."""
        entry = SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello world")
        expected = "5\n00:00:01,000 --> 00:00:03,500\nHello world\n\n"
        assert entry.to_srt(new_idx=5) == expected
