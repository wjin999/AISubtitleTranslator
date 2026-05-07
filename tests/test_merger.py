"""Tests for subtitle merging module."""

import pytest
from srt_translator.models import SrtEntry
from srt_translator.merger import should_merge, merge_entries, init_spacy_model

try:
    import spacy
    spacy.load("en_core_web_sm")
    spacy_available = True
except (ImportError, OSError):
    spacy_available = False


class TestShouldMergeBasic:
    """不依赖 spaCy 的基础合并判断测试，始终运行。"""
    
    def test_time_gap_too_large(self):
        cur = SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello")
        nxt = SrtEntry(2, "00:00:10,000", "00:00:13,000", "world")
        result = should_merge(cur, nxt, 300, 1.5)
        assert result is False  # 时间间隔 6.5s > 1.5s
    
    def test_max_chars_exceeded(self):
        cur = SrtEntry(1, "00:00:01,000", "00:00:03,500", "A" * 200)
        nxt = SrtEntry(2, "00:00:03,600", "00:00:06,000", "B" * 200)
        result = should_merge(cur, nxt, 300, 1.5)
        assert result is False  # 总长 > 300
    
    def test_sentence_ending_punctuation(self):
        cur = SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello.")
        nxt = SrtEntry(2, "00:00:03,600", "00:00:06,000", "world")
        result = should_merge(cur, nxt, 300, 1.5)
        assert result is False  # 句尾有句号
    
    def test_empty_text(self):
        cur = SrtEntry(1, "00:00:01,000", "00:00:03,500", "")
        nxt = SrtEntry(2, "00:00:03,600", "00:00:06,000", "world")
        result = should_merge(cur, nxt, 300, 1.5)
        assert result is False


@pytest.mark.skipif(not spacy_available, reason="spaCy en_core_web_sm not installed")
class TestShouldMergeNLP:
    """依赖 spaCy NLP 的合并判断测试。"""
    
    def setup_method(self):
        """在每个测试前初始化 spaCy 模型。"""
        init_spacy_model()
    
    def test_sentence_boundary_detection(self):
        # 以下测试需要 NLP 检测句子边界
        cur = SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello world")
        nxt = SrtEntry(2, "00:00:03,600", "00:00:06,000", "how are you")
        result = should_merge(cur, nxt, 300, 1.5)
        assert result is True  # 非完整句子结尾，应合并


class TestMergeEntries:
    
    def test_empty_list(self):
        assert merge_entries([]) == []
    
    def test_single_entry(self):
        entries = [SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello")]
        merged = merge_entries(entries)
        assert len(merged) == 1
        assert merged[0].text == "Hello"
    
    def test_no_merge_due_to_original_unmodified(self):
        """验证 merge_entries 不修改原始 entries"""
        entries = [
            SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello"),
            SrtEntry(2, "00:00:10,000", "00:00:13,000", "world"),
        ]
        original_text = entries[0].text
        merge_entries(entries)
        assert entries[0].text == original_text  # 未被修改
