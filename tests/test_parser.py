"""Tests for SRT parser."""

import pytest
from pathlib import Path
import tempfile

from srt_translator.parser import parse_srt, save_srt
from srt_translator.models import SrtEntry


class TestParseSrt:
    """Test cases for parse_srt function."""
    
    def test_parse_simple(self):
        """Test parsing simple SRT content."""
        content = """1
00:00:01,000 --> 00:00:03,500
Hello world

2
00:00:04,000 --> 00:00:06,500
Goodbye world

"""
        entries = parse_srt(content)
        assert len(entries) == 2
        assert entries[0].index == 1
        assert entries[0].text == "Hello world"
        assert entries[1].index == 2
        assert entries[1].text == "Goodbye world"
    
    def test_parse_multiline_text(self):
        """Test parsing multiline subtitle text."""
        content = """1
00:00:01,000 --> 00:00:03,500
Line one
Line two
Line three

"""
        entries = parse_srt(content)
        assert len(entries) == 1
        assert entries[0].text == "Line one Line two Line three"
    
    def test_parse_empty_content(self):
        """Test parsing empty content."""
        entries = parse_srt("")
        assert len(entries) == 0
    
    def test_parse_preserves_timecodes(self):
        """Test that timecodes are preserved correctly."""
        content = """1
01:30:45,500 --> 01:30:50,250
Test

"""
        entries = parse_srt(content)
        assert entries[0].start == "01:30:45,500"
        assert entries[0].end == "01:30:50,250"


class TestSaveSrt:
    """Test cases for save_srt function."""
    
    def test_save_and_reload(self):
        """Test saving and reloading SRT file."""
        entries = [
            SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello"),
            SrtEntry(2, "00:00:04,000", "00:00:06,500", "World"),
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            path = Path(f.name)
        
        try:
            save_srt(entries, path)
            content = path.read_text(encoding='utf-8')
            reloaded = parse_srt(content)
            
            assert len(reloaded) == 2
            assert reloaded[0].text == "Hello"
            assert reloaded[1].text == "World"
        finally:
            path.unlink()
    
    def test_save_reindexes(self):
        """Test that save_srt reindexes entries."""
        entries = [
            SrtEntry(10, "00:00:01,000", "00:00:03,500", "First"),
            SrtEntry(20, "00:00:04,000", "00:00:06,500", "Second"),
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            path = Path(f.name)
        
        try:
            save_srt(entries, path)
            content = path.read_text(encoding='utf-8')
            reloaded = parse_srt(content)
            
            assert reloaded[0].index == 1
            assert reloaded[1].index == 2
        finally:
            path.unlink()
