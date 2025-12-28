"""Tests for text utilities."""

import pytest
from srt_translator.text_utils import clean_translated_text, validate_translation


class TestCleanTranslatedText:
    
    def test_remove_list_markers(self):
        assert clean_translated_text("1. Hello") == "Hello"
        assert clean_translated_text("- Hello") == "Hello"
        assert clean_translated_text("* Hello") == "Hello"
    
    def test_remove_markdown(self):
        assert clean_translated_text("**bold**") == "bold"
        assert clean_translated_text("*italic*") == "italic"
    
    def test_preserve_chinese_punctuation(self):
        text = "你好，世界！这是测试。"
        # Should not remove Chinese punctuation
        result = clean_translated_text(text)
        assert "，" in result
        assert "！" in result
        assert "。" in result
    
    def test_normalize_whitespace(self):
        assert clean_translated_text("  hello   world  ") == "hello world"
    
    def test_empty_input(self):
        assert clean_translated_text("") == ""
        assert clean_translated_text(None) == ""


class TestValidateTranslation:
    
    def test_valid(self):
        is_valid, error = validate_translation("Hello world", "你好世界")
        assert is_valid
        assert error == ""
    
    def test_empty(self):
        is_valid, error = validate_translation("Hello", "")
        assert not is_valid
        assert "Empty" in error
    
    def test_fail_marker(self):
        is_valid, error = validate_translation("Hello", "[Fail] Hello")
        assert not is_valid
        assert "failed" in error
    
    def test_too_short(self):
        is_valid, error = validate_translation("This is a long sentence", "X")
        assert not is_valid
        assert "too short" in error
    
    def test_identical(self):
        is_valid, error = validate_translation("Hello", "Hello")
        assert not is_valid
        assert "identical" in error
