"""Intelligent subtitle merging using spaCy NLP."""

from __future__ import annotations

import logging
from typing import List, Sequence, Optional, TYPE_CHECKING

from .models import SrtEntry

if TYPE_CHECKING:
    import spacy

logger = logging.getLogger(__name__)

# Global NLP model instance
_nlp_model: Optional["spacy.Language"] = None


def init_spacy_model() -> None:
    """
    Initialize the spaCy NLP model.
    
    Downloads the model if not available.
    """
    global _nlp_model
    
    if _nlp_model is not None:
        return  # 已初始化
    
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy is required for smart merging. "
            "Install it with 'pip install spacy' or use --no-merge flag."
        )
    
    logger.info("Initializing spaCy NLP model...")
    
    try:
        _nlp_model = spacy.load(
            "en_core_web_sm", 
            disable=["ner", "textcat", "lemmatizer"]
        )
        logger.info("spaCy model loaded successfully")
    except OSError:
        logger.warning("Model 'en_core_web_sm' not found. Downloading...")
        from spacy.cli import download
        download("en_core_web_sm")
        _nlp_model = spacy.load(
            "en_core_web_sm", 
            disable=["ner", "textcat", "lemmatizer"]
        )


def _check_sentence_boundary(text1: str, text2: str) -> bool:
    """
    Check if combining two texts would cross a sentence boundary.
    
    Returns:
        True if they should be merged (no boundary between them)
    """
    global _nlp_model
    
    if _nlp_model is None:
        raise RuntimeError("spaCy model not initialized")
    
    combined = f"{text1} {text2}"
    doc = _nlp_model(combined)
    sentences = list(doc.sents)
    
    # 如果只有一个句子，可以合并
    if len(sentences) == 1:
        return True
    
    # 检查句子边界是否在拼接点附近
    split_point = len(text1) + 1  # +1 for space
    
    for sent in sentences:
        # 如果句子边界在拼接点附近（±2字符），不应合并
        if abs(sent.start_char - split_point) <= 2:
            return False
    
    return True


def should_merge(
    cur_entry: SrtEntry, 
    next_entry: SrtEntry, 
    max_chars: int,
    time_gap_threshold: float
) -> bool:
    """
    Determine if two subtitle entries should be merged.
    """
    cur_text = cur_entry.text
    next_text = next_entry.text
    
    # 基础检查
    if not cur_text or not next_text:
        return False
    
    # 长度检查
    if len(cur_text) + 1 + len(next_text) > max_chars:
        return False
    
    # 时间间隔检查
    time_gap = next_entry.start_seconds - cur_entry.end_seconds
    if time_gap > time_gap_threshold:
        return False
    
    # 如果当前字幕以句号等结尾，不合并
    if cur_text.rstrip()[-1:] in '.!?。！？':
        return False

    # NLP 句子边界检测
    return _check_sentence_boundary(cur_text, next_text)


def merge_entries(
    entries: Sequence[SrtEntry], 
    max_chars: int = 300,
    time_gap_threshold: float = 1.5
) -> List[SrtEntry]:
    """
    Merge subtitle entries using intelligent NLP-based logic.
    
    注意：此函数不会修改原始 entries，而是返回新的列表。
    """
    if not entries:
        return []
    
    if len(entries) == 1:
        return [entries[0].copy()]
    
    merged: List[SrtEntry] = []
    
    # 创建第一个条目的副本作为当前合并目标
    current = entries[0].copy()
    
    for i in range(1, len(entries)):
        next_entry = entries[i]
        
        if should_merge(current, next_entry, max_chars, time_gap_threshold):
            # 合并：更新当前条目（不修改原始对象）
            current = current.copy(
                text=current.text + " " + next_entry.text,
                end=next_entry.end
            )
        else:
            # 不合并：保存当前条目，开始新的合并
            merged.append(current)
            current = next_entry.copy()
    
    # 添加最后一个条目
    merged.append(current)
    
    logger.info(f"Merged {len(entries)} entries into {len(merged)} entries")
    return merged


def merge_entries_batch(
    entries: Sequence[SrtEntry], 
    max_chars: int = 300,
    time_gap_threshold: float = 1.5,
    batch_size: int = 100
) -> List[SrtEntry]:
    """
    Batch-optimized version of merge_entries.
    
    使用 spaCy 的 nlp.pipe() 批量处理以提高性能。
    适用于大量字幕条目。
    """
    global _nlp_model
    
    if not entries:
        return []
    
    if _nlp_model is None:
        raise RuntimeError("spaCy model not initialized")
    
    # 对于小数据集，使用普通方法
    if len(entries) < batch_size:
        return merge_entries(entries, max_chars, time_gap_threshold)
    
    # 预计算所有相邻对的合并文本
    pairs_to_check: List[tuple[int, str]] = []
    
    for i in range(len(entries) - 1):
        cur = entries[i]
        nxt = entries[i + 1]
        
        # 先做快速检查
        if not cur.text or not nxt.text:
            continue
        if len(cur.text) + 1 + len(nxt.text) > max_chars:
            continue
        if nxt.start_seconds - cur.end_seconds > time_gap_threshold:
            continue
        if cur.text.rstrip()[-1:] in '.!?。！？':
            continue
        
        # 需要 NLP 检查的对
        combined = f"{cur.text} {nxt.text}"
        pairs_to_check.append((i, combined))
    
    # 批量 NLP 处理
    should_merge_set: set[int] = set()
    
    if pairs_to_check:
        texts = [p[1] for p in pairs_to_check]
        docs = list(_nlp_model.pipe(texts, batch_size=batch_size))
        
        for (idx, _), doc in zip(pairs_to_check, docs):
            sentences = list(doc.sents)
            if len(sentences) == 1:
                should_merge_set.add(idx)
            else:
                # 检查边界
                split_point = len(entries[idx].text) + 1
                merge_ok = True
                for sent in sentences:
                    if abs(sent.start_char - split_point) <= 2:
                        merge_ok = False
                        break
                if merge_ok:
                    should_merge_set.add(idx)
    
    # 执行合并
    merged: List[SrtEntry] = []
    current = entries[0].copy()
    
    for i in range(1, len(entries)):
        if i - 1 in should_merge_set:
            current = current.copy(
                text=current.text + " " + entries[i].text,
                end=entries[i].end
            )
        else:
            merged.append(current)
            current = entries[i].copy()
    
    merged.append(current)
    
    logger.info(f"Batch merged {len(entries)} entries into {len(merged)} entries")
    return merged
