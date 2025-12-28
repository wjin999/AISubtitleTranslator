"""Progress tracking and resume support."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TranslationProgress:
    """翻译进度记录。"""
    
    input_file: str
    total_chunks: int
    completed_chunks: List[int]
    translations: Dict[int, str]  # index -> translated text
    started_at: str
    updated_at: str
    
    @classmethod
    def create(cls, input_file: str, total_chunks: int) -> "TranslationProgress":
        """创建新的进度记录。"""
        now = datetime.now().isoformat()
        return cls(
            input_file=input_file,
            total_chunks=total_chunks,
            completed_chunks=[],
            translations={},
            started_at=now,
            updated_at=now,
        )
    
    def mark_completed(self, chunk_idx: int, results: Dict[int, str]) -> None:
        """标记 chunk 完成并保存翻译结果。"""
        if chunk_idx not in self.completed_chunks:
            self.completed_chunks.append(chunk_idx)
        self.translations.update(results)
        self.updated_at = datetime.now().isoformat()
    
    @property
    def is_complete(self) -> bool:
        """检查是否全部完成。"""
        return len(self.completed_chunks) >= self.total_chunks
    
    @property
    def completion_rate(self) -> float:
        """完成率 (0-1)。"""
        if self.total_chunks == 0:
            return 1.0
        return len(self.completed_chunks) / self.total_chunks
    
    def get_pending_chunks(self) -> List[int]:
        """获取未完成的 chunk 索引列表。"""
        return [i for i in range(self.total_chunks) if i not in self.completed_chunks]


def get_progress_file(input_path: Path) -> Path:
    """获取进度文件路径。"""
    return input_path.with_suffix(input_path.suffix + ".progress.json")


def save_progress(progress: TranslationProgress, path: Path) -> bool:
    """
    保存进度到文件。
    
    Returns:
        True if successful
    """
    try:
        data = asdict(progress)
        # 将 int keys 转换为 str (JSON 要求)
        data['translations'] = {str(k): v for k, v in data['translations'].items()}
        
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Progress saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")
        return False


def load_progress(path: Path) -> Optional[TranslationProgress]:
    """
    从文件加载进度。
    
    Returns:
        TranslationProgress if found and valid, None otherwise
    """
    if not path.exists():
        return None
    
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 将 str keys 转换回 int
        data['translations'] = {int(k): v for k, v in data['translations'].items()}
        
        return TranslationProgress(**data)
    except Exception as e:
        logger.warning(f"Failed to load progress file: {e}")
        return None


def delete_progress(path: Path) -> None:
    """删除进度文件。"""
    try:
        if path.exists():
            path.unlink()
            logger.debug(f"Progress file deleted: {path}")
    except Exception as e:
        logger.warning(f"Failed to delete progress file: {e}")
