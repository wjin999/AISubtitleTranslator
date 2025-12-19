"""Glossary loading and management utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def load_glossary(path: Path) -> Dict[str, str]:
    """
    Load glossary from a text file.
    
    Glossary file format:
        Term = Translation
        Another Term = Another Translation
    
    Args:
        path: Path to glossary file
        
    Returns:
        Dictionary mapping source terms to translations
    """
    glossary: Dict[str, str] = {}
    
    if not path.exists():
        logger.warning(f"Glossary file not found: {path}")
        return glossary
    
    try:
        content = path.read_text(encoding='utf-8')
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                term, trans = line.split('=', 1)
                term = term.strip()
                trans = trans.strip()
                if term and trans:
                    glossary[term] = trans
                else:
                    logger.warning(f"Invalid glossary entry at line {line_num}: {line}")
            else:
                logger.warning(f"Skipping invalid line {line_num} (no '=' found): {line}")
    except Exception as e:
        logger.error(f"Error loading glossary: {e}")
    
    logger.info(f"Loaded {len(glossary)} terms from glossary.")
    return glossary


def find_matching_terms(glossary: Dict[str, str], text: str) -> Dict[str, str]:
    """
    Find glossary terms that appear in the given text.
    
    Args:
        glossary: Full glossary dictionary
        text: Text to search for terms
        
    Returns:
        Dictionary of matched terms and their translations
    """
    text_lower = text.lower()
    return {
        term: trans
        for term, trans in glossary.items()
        if term.lower() in text_lower
    }
