"""Tests for glossary module."""

import pytest
from pathlib import Path
import tempfile

from srt_translator.glossary import Glossary, load_glossary, find_matching_terms


class TestGlossary:
    
    def test_add_and_get(self):
        g = Glossary()
        g.add("Hello", "你好")
        assert g.get("Hello") == "你好"
    
    def test_case_insensitive_get(self):
        g = Glossary()
        g.add("Hello", "你好")
        assert g.get("hello") == "你好"
        assert g.get("HELLO") == "你好"
    
    def test_find_matches(self):
        g = Glossary()
        g.add("Hello", "你好")
        g.add("World", "世界")
        g.add("Goodbye", "再见")
        
        matches = g.find_matches("Hello World!")
        assert "Hello" in matches
        assert "World" in matches
        assert "Goodbye" not in matches
    
    def test_len_and_bool(self):
        g = Glossary()
        assert len(g) == 0
        assert not g
        
        g.add("Test", "测试")
        assert len(g) == 1
        assert g


class TestLoadGlossary:
    
    def test_load_simple(self):
        content = """Hello = 你好
World = 世界
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            path = Path(f.name)
        
        try:
            glossary = load_glossary(path)
            assert glossary.get("Hello") == "你好"
            assert glossary.get("World") == "世界"
        finally:
            path.unlink()
    
    def test_load_with_arrow_separator(self):
        content = "Hello -> 你好\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            path = Path(f.name)
        
        try:
            glossary = load_glossary(path)
            assert glossary.get("Hello") == "你好"
        finally:
            path.unlink()
    
    def test_load_with_comments(self):
        content = """# Comment
Hello = 你好
# Another comment
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            path = Path(f.name)
        
        try:
            glossary = load_glossary(path)
            assert len(glossary) == 1
        finally:
            path.unlink()
    
    def test_nonexistent(self):
        glossary = load_glossary(Path("/nonexistent"))
        assert len(glossary) == 0


class TestFindMatchingTerms:
    
    def test_with_glossary_class(self):
        g = Glossary()
        g.add("Hello", "你好")
        
        matches = find_matching_terms(g, "Hello there!")
        assert matches == {"Hello": "你好"}
    
    def test_with_dict(self):
        d = {"Hello": "你好", "World": "世界"}
        
        matches = find_matching_terms(d, "Hello World")
        assert "Hello" in matches
        assert "World" in matches
