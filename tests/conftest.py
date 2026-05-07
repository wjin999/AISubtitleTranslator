"""Shared fixtures and configurations for tests."""

import pytest


@pytest.fixture(scope="session")
def spacy_available():
    """Check if spaCy model en_core_web_sm is available."""
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except (ImportError, OSError):
        return False
