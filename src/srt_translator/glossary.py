"""Glossary loading and management utilities."""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class Glossary:
    """术语表管理类，支持精确匹配和模糊匹配。"""
    
    def __init__(self):
        # 保存原始大小写的术语
        self._terms: Dict[str, str] = {}
        # 小写索引用于匹配
        self._lower_index: Dict[str, str] = {}
    
    def add(self, term: str, translation: str) -> None:
        """添加术语。"""
        term = term.strip()
        translation = translation.strip()
        if term and translation:
            self._terms[term] = translation
            self._lower_index[term.lower()] = term
    
    def get(self, term: str) -> str | None:
        """获取术语翻译（不区分大小写）。"""
        original_term = self._lower_index.get(term.lower())
        if original_term:
            return self._terms.get(original_term)
        return None
    
    def find_matches(self, text: str) -> Dict[str, str]:
        """
        在文本中查找匹配的术语。
        
        返回原始大小写的术语及其翻译。
        """
        text_lower = text.lower()
        matches: Dict[str, str] = {}
        
        for term, translation in self._terms.items():
            if term.lower() in text_lower:
                matches[term] = translation
        
        return matches
    
    def __len__(self) -> int:
        return len(self._terms)
    
    def __bool__(self) -> bool:
        return len(self._terms) > 0


def load_glossary(path: Path) -> Glossary:
    """
    Load glossary from a text file.
    
    Supported formats:
        Term = Translation
        Term -> Translation
        # Comment lines
    
    Args:
        path: Path to glossary file
        
    Returns:
        Glossary instance
    """
    glossary = Glossary()
    
    if not path.exists():
        logger.warning(f"Glossary file not found: {path}")
        return glossary
    
    try:
        content = path.read_text(encoding='utf-8')
        
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            # 支持 = 和 -> 两种分隔符
            match = re.match(r'^(.+?)\s*(?:=|->)\s*(.+)$', line)
            if match:
                term, translation = match.groups()
                glossary.add(term, translation)
            else:
                logger.debug(f"Skipping invalid line {line_num}: {line}")
                
    except Exception as e:
        logger.error(f"Error loading glossary: {e}")
    
    logger.info(f"Loaded {len(glossary)} terms from glossary")
    return glossary


def find_matching_terms(glossary: Glossary | Dict[str, str], text: str) -> Dict[str, str]:
    """
    Find glossary terms that appear in the given text.
    
    兼容旧的 Dict 接口和新的 Glossary 类。
    """
    if isinstance(glossary, Glossary):
        return glossary.find_matches(text)
    
    # 兼容旧的 Dict 接口
    text_lower = text.lower()
    return {
        term: trans
        for term, trans in glossary.items()
        if term.lower() in text_lower
    }
