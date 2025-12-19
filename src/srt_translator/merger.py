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
    
    Raises:
        ImportError: If spaCy is not installed
    """
    global _nlp_model
    
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
    except OSError:
        logger.warning("Model 'en_core_web_sm' not found. Downloading...")
        from spacy.cli import download
        download("en_core_web_sm")
        _nlp_model = spacy.load(
            "en_core_web_sm", 
            disable=["ner", "textcat", "lemmatizer"]
        )


def should_merge(
    cur_entry: SrtEntry, 
    next_entry: SrtEntry, 
    max_chars: int,
    time_gap_threshold: float
) -> bool:
    """
    Determine if two subtitle entries should be merged.
    
    Uses spaCy sentence boundary detection to make intelligent decisions.
    
    Args:
        cur_entry: Current subtitle entry
        next_entry: Next subtitle entry
        max_chars: Maximum character limit for merged text
        time_gap_threshold: Maximum time gap (seconds) to allow merging
        
    Returns:
        True if entries should be merged, False otherwise
    """
    global _nlp_model
    
    if _nlp_model is None:
        raise RuntimeError("spaCy model not initialized. Call init_spacy_model() first.")
    
    cur_text = cur_entry.text
    next_text = next_entry.text
    
    # Basic checks
    if not cur_text or not next_text:
        return False
    if len(cur_text) + 1 + len(next_text) > max_chars:
        return False
    
    # Time gap check
    time_gap = next_entry.start_seconds - cur_entry.end_seconds
    if time_gap > time_gap_threshold:
        return False

    # NLP-based sentence boundary detection
    combined_text = f"{cur_text} {next_text}"
    doc = _nlp_model(combined_text)
    sentences = list(doc.sents)
    
    # If combined text forms a single sentence, merge
    if len(sentences) == 1:
        return True
    
    # Check if sentence boundary aligns with the split point
    split_index = len(cur_text) + 1  # +1 for the space
    for sent in sentences:
        if abs(sent.start_char - split_index) <= 2:
            # Sentence boundary aligns with split - don't merge
            return False
    
    return True


def merge_entries(
    entries: Sequence[SrtEntry], 
    max_chars: int = 300,
    time_gap_threshold: float = 1.5
) -> List[SrtEntry]:
    """
    Merge subtitle entries using intelligent NLP-based logic.
    
    Args:
        entries: Sequence of SrtEntry objects to merge
        max_chars: Maximum characters per merged entry
        time_gap_threshold: Maximum time gap (seconds) between entries to merge
        
    Returns:
        List of merged SrtEntry objects
    """
    if not entries:
        return []
    
    merged: List[SrtEntry] = []
    current = entries[0]
    
    for i in range(1, len(entries)):
        next_entry = entries[i]
        if should_merge(current, next_entry, max_chars, time_gap_threshold):
            # Merge entries
            current.text += " " + next_entry.text
            current.end = next_entry.end
        else:
            merged.append(current)
            current = next_entry
    
    merged.append(current)
    return merged
