"""Progress tracking and resume support."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
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
    
    # 缓存本次 mark_completed 新增的翻译结果（用于增量保存）
    _last_chunk_results: Dict[int, str] = field(default_factory=dict, repr=False)
    
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
    
    def mark_completed(self, chunk_idx: int, results: Dict[int, str]) -> Dict[int, str]:
        """
        标记 chunk 完成并保存翻译结果。
        
        Returns:
            本次新增的翻译结果 dict（用于增量保存）
        """
        if chunk_idx not in self.completed_chunks:
            self.completed_chunks.append(chunk_idx)
        
        # 只提取本次真正新增的条目（避免重复保存已存在的）
        new_results = {k: v for k, v in results.items() if k not in self.translations}
        self.translations.update(new_results)
        self.updated_at = datetime.now().isoformat()
        
        # 缓存用于增量保存
        self._last_chunk_results = new_results
        
        return new_results  # 返回增量，供外部只保存新增部分
    
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


def save_progress_incremental(
    progress: TranslationProgress,
    path: Path,
    new_results: Dict[int, str],
    chunk_idx: int | None = None,  # 新增参数：用于在 NDJSON 中记录 chunk 索引
) -> bool:
    """
    增量保存进度：只追加新增的翻译结果，不重写整个文件。
    使用 NDJSON 格式（每行一个 JSON 对象），避免每次重写整个文件。
    
    加载时通过 load_progress() 合并所有增量行。
    
    文件格式：
    {"type":"meta","input_file":"...","total_chunks":10,"started_at":"..."}
    {"type":"chunk","chunk_idx":0,"updated_at":"...","translations":{"0":"你好","1":"世界"}}
    {"type":"chunk","chunk_idx":1,"updated_at":"...","translations":{"2":"测试"}}
    
    Args:
        progress: 进度对象
        path: 进度文件路径
        new_results: 本次新增的翻译结果 {index: text}
        chunk_idx: 可选，当前 chunk 的索引（用于精确恢复 completed_chunks）
    
    Returns:
        True if successful
    """
    try:
        # 确保 meta 信息已存在（先检查再写入，减少竞态窗口）
        is_new = not path.exists() or path.stat().st_size == 0
        if is_new:
            path.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "type": "meta",
                "input_file": progress.input_file,
                "total_chunks": progress.total_chunks,
                "started_at": progress.started_at,
            }
            with path.open('w', encoding='utf-8') as f:
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        
        # 追加写入 chunk 数据
        if new_results:
            str_results = {str(k): v for k, v in new_results.items()}
            data = {
                "type": "chunk",
                "chunk_idx": chunk_idx,
                "updated_at": progress.updated_at,
                "translations": str_results,
            }
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        return True
    except Exception as e:
        logger.error(f"Failed to save progress incremental: {e}")
        return False


def load_progress(path: Path) -> Optional[TranslationProgress]:
    """
    从文件加载进度（同时支持旧的全量 JSON 和新的 NDJSON 格式）。
    
    Returns:
        TranslationProgress if found and valid, None otherwise
    """
    if not path.exists():
        return None
    
    try:
        content = path.read_text(encoding='utf-8').strip()
        if not content:
            return None
        
        # 尝试解析为 NDJSON（多行）
        lines = content.split('\n')
        
        if len(lines) == 1:
            # 旧格式：单行全量 JSON
            data = json.loads(lines[0])
            data['translations'] = {int(k): v for k, v in data['translations'].items()}
            # 移除可能的内部字段
            data.pop('_last_chunk_results', None)
            return TranslationProgress(**data)
        else:
            # 新格式：NDJSON，逐行合并
            meta_line = json.loads(lines[0])
            if meta_line.get('type') != 'meta':
                logger.warning("Invalid NDJSON progress file: first line is not meta")
                return None
            
            translations: Dict[int, str] = {}
            completed_chunks: List[int] = []
            last_updated = meta_line.get('started_at', '')
            
            for line in lines[1:]:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get('type') == 'chunk':
                    chunk_translations = data.get('translations', {})
                    for k, v in chunk_translations.items():
                        translations[int(k)] = v
                    last_updated = data.get('updated_at', last_updated)
                    # 直接从 chunk 行恢复 chunk_idx，避免推断
                    cidx = data.get('chunk_idx')
                    if cidx is not None:
                        completed_chunks.append(cidx)
            
            total_chunks = meta_line.get('total_chunks', 0)
            
            progress = TranslationProgress(
                input_file=meta_line.get('input_file', ''),
                total_chunks=total_chunks,
                completed_chunks=completed_chunks,
                translations=translations,
                started_at=meta_line.get('started_at', ''),
                updated_at=last_updated,
            )
            return progress
            
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
