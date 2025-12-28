"""Tests for SrtEntry model."""

import pytest
from srt_translator.models import SrtEntry


class TestSrtEntry:
    
    def test_creation(self):
        entry = SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello world")
        assert entry.index == 1
        assert entry.text == "Hello world"
    
    def test_timecode_property(self):
        entry = SrtEntry(1, "00:00:01,000", "00:00:03,500", "Test")
        assert entry.timecode == "00:00:01,000 --> 00:00:03,500"
    
    def test_start_seconds_cached(self):
        entry = SrtEntry(1, "01:30:45,500", "01:30:50,000", "Test")
        # First call
        result1 = entry.start_seconds
        # Second call should use cache
        result2 = entry.start_seconds
        assert result1 == result2 == 5445.5
    
    def test_invalid_timecode(self):
        entry = SrtEntry(1, "invalid", "00:00:01,000", "Test")
        assert entry.start_seconds == 0.0
    
    def test_to_srt(self):
        entry = SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello")
        expected = "1\n00:00:01,000 --> 00:00:03,500\nHello\n\n"
        assert entry.to_srt() == expected
    
    def test_copy(self):
        entry = SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello")
        copied = entry.copy(text="World", end="00:00:05,000")
        
        # Original unchanged
        assert entry.text == "Hello"
        assert entry.end == "00:00:03,500"
        
        # Copy has new values
        assert copied.text == "World"
        assert copied.end == "00:00:05,000"
        assert copied.start == entry.start  # Unchanged field
