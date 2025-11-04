"""
[DEMO MODE] Mock APIs Module
This module provides mock implementations of external APIs for demonstration purposes.
Replace these with real API calls in production.
"""

from .mock_openai import MockOpenAI
from .mock_gemini import MockGemini
from .mock_vision import MockVisionExtractor

__all__ = ["MockOpenAI", "MockGemini", "MockVisionExtractor"]
