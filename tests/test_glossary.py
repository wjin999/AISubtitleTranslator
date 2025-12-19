"""Tests for glossary module."""

import pytest
from pathlib import Path
import tempfile

from srt_translator.glossary import load_glossary, find_matching_terms


class TestLoadGlossary:
    """Test cases for load_glossary function."""
    
    def test_load_simple_glossary(self):
        """Test loading a simple glossary file."""
        content = """Term1 = Translation1
Term2 = Translation2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            path = Path(f.name)
        
        try:
            glossary = load_glossary(path)
            assert glossary["Term1"] == "Translation1"
            assert glossary["Term2"] == "Translation2"
        finally:
            path.unlink()
    
    def test_load_with_comments(self):
        """Test loading glossary with comments."""
        content = """# This is a comment
Term1 = Translation1
# Another comment
Term2 = Translation2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            path = Path(f.name)
        
        try:
            glossary = load_glossary(path)
            assert len(glossary) == 2
            assert "# This is a comment" not in glossary
        finally:
            path.unlink()
    
    def test_load_with_empty_lines(self):
        """Test loading glossary with empty lines."""
        content = """Term1 = Translation1

Term2 = Translation2

"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            path = Path(f.name)
        
        try:
            glossary = load_glossary(path)
            assert len(glossary) == 2
        finally:
            path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        path = Path("/nonexistent/glossary.txt")
        glossary = load_glossary(path)
        assert glossary == {}
    
    def test_load_with_whitespace(self):
        """Test loading glossary with extra whitespace."""
        content = """  Term1   =   Translation1  
Term2=Translation2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            path = Path(f.name)
        
        try:
            glossary = load_glossary(path)
            assert glossary["Term1"] == "Translation1"
            assert glossary["Term2"] == "Translation2"
        finally:
            path.unlink()


class TestFindMatchingTerms:
    """Test cases for find_matching_terms function."""
    
    def test_find_matching(self):
        """Test finding matching terms."""
        glossary = {
            "Hello": "你好",
            "World": "世界",
            "Goodbye": "再见",
        }
        text = "Hello World!"
        matched = find_matching_terms(glossary, text)
        
        assert "Hello" in matched
        assert "World" in matched
        assert "Goodbye" not in matched
    
    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        glossary = {"Hello": "你好"}
        text = "hello there"
        matched = find_matching_terms(glossary, text)
        
        assert "Hello" in matched
    
    def test_no_matches(self):
        """Test when no terms match."""
        glossary = {"Hello": "你好"}
        text = "Goodbye World"
        matched = find_matching_terms(glossary, text)
        
        assert len(matched) == 0
