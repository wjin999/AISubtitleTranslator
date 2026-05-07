"""Tests for progress tracking module."""

import pytest
import tempfile
from pathlib import Path
from srt_translator.progress import (
    TranslationProgress,
    save_progress_incremental,
    load_progress,
    delete_progress,
)


class TestTranslationProgress:
    
    def test_create(self):
        p = TranslationProgress.create("test.srt", 10)
        assert p.total_chunks == 10
        assert p.completed_chunks == []
        assert p.translations == {}
        assert p.completion_rate == 0.0
    
    def test_mark_completed(self):
        p = TranslationProgress.create("test.srt", 10)
        results = {0: "你好", 1: "世界"}
        new_results = p.mark_completed(0, results)
        
        assert 0 in p.completed_chunks
        assert p.translations[0] == "你好"
        assert new_results == {0: "你好", 1: "世界"}
    
    def test_completion_rate(self):
        p = TranslationProgress.create("test.srt", 10)
        assert p.completion_rate == 0.0
        p.mark_completed(0, {})
        assert p.completion_rate == 0.1
        for i in range(1, 10):
            p.mark_completed(i, {})
        assert p.completion_rate == 1.0
        assert p.is_complete
    
    def test_get_pending(self):
        p = TranslationProgress.create("test.srt", 5)
        assert p.get_pending_chunks() == [0, 1, 2, 3, 4]
        p.mark_completed(1, {})
        p.mark_completed(3, {})
        assert p.get_pending_chunks() == [0, 2, 4]


class TestProgressIO:
    
    def test_save_and_load_incremental(self):
        p = TranslationProgress.create("test.srt", 5)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)
        
        try:
            # 第一次写入
            new_results = p.mark_completed(0, {0: "你好"})
            save_progress_incremental(p, path, new_results, chunk_idx=0)
            
            # 第二次写入
            new_results = p.mark_completed(1, {1: "世界"})
            save_progress_incremental(p, path, new_results, chunk_idx=1)
            
            # 重新加载
            loaded = load_progress(path)
            assert loaded is not None
            assert loaded.total_chunks == 5
            assert loaded.translations[0] == "你好"
            assert loaded.translations[1] == "世界"
            assert 0 in loaded.completed_chunks
            assert 1 in loaded.completed_chunks
        finally:
            path.unlink()
    
    def test_load_nonexistent(self):
        assert load_progress(Path("/nonexistent.json")) is None
    
    def test_delete_progress(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)
            path.write_text("test")
        
        assert path.exists()
        delete_progress(path)
        assert not path.exists()
