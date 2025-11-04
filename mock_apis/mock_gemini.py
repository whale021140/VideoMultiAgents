"""
[MOCK API] - Mock Google Gemini API Implementation
This module simulates Google Gemini API responses for demonstration purposes.
Replace with real google.generativeai client in production.

Usage in production:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-pro-vision')
"""

from typing import Dict, List, Any, Optional


class MockGenerateResponse:
    """[MOCK API] - Represents a mock Gemini API response"""
    def __init__(self, text: str):
        self.text = text


class MockGemini:
    """
    [MOCK API] - Mock Gemini client for demonstration
    Returns predefined responses for testing without API keys
    
    Replacement instructions for production:
    1. Replace with: import google.generativeai as genai
    2. Configure: genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    3. Initialize: model = genai.GenerativeModel('gemini-pro-vision')
    4. Ensure environment variable GEMINI_API_KEY is set
    """
    
    def __init__(self, api_key: str = "mock_key_67890"):
        """
        [MOCK API] Initialize mock Gemini client
        In production: genai.GenerativeModel('gemini-pro-vision')
        """
        self.api_key = api_key
        self.model_name = "gemini-pro-vision"
    
    def generate_content(self, content: Any, **kwargs) -> MockGenerateResponse:
        """
        [MOCK API] - Mock Gemini content generation
        
        In production:
            response = model.generate_content(content)
            return response.text
        
        Args:
            content: Input content (text or image data)
            **kwargs: Additional generation parameters
        
        Returns:
            MockGenerateResponse with mock text response
        """
        # [MOCK DATA] - Predefined responses for common scenarios
        mock_responses = {
            "image_analysis": "Image Analysis Result: [DEMO] The image shows typical scene elements. Identified objects include foreground, background, and contextual elements. Color distribution shows natural variations. Overall composition suggests: [DEMO RESPONSE]",
            "text_analysis": "Text Analysis Result: [DEMO] The provided text contains key information about the subject matter. Main themes identified: 1) Primary concept, 2) Secondary elements, 3) Context. Sentiment analysis shows neutral-to-positive tone. Key insight: [DEMO RESPONSE]",
            "multimodal": "Multimodal Analysis Result: [DEMO] Combining visual and textual information, the comprehensive understanding is: Visual elements support [DEMO], Text indicates [DEMO], Combined conclusion: [DEMO RESPONSE]",
            "default": "[DEMO] Mock Gemini response generated for testing. Replace with real API for production."
        }
        
        # Determine response type based on content
        content_str = str(content).lower()
        selected_response = mock_responses["default"]
        
        if "image" in content_str:
            selected_response = mock_responses["image_analysis"]
        elif "text" in content_str:
            selected_response = mock_responses["text_analysis"]
        elif "multimodal" in content_str or ("image" in content_str and "text" in content_str):
            selected_response = mock_responses["multimodal"]
        
        # [MOCK API] Return mock response
        return MockGenerateResponse(selected_response)
    
    def count_tokens(self, content: str) -> Dict[str, int]:
        """
        [MOCK API] - Mock token counting
        
        In production:
            response = model.count_tokens(content)
            return {"total_tokens": response.total_tokens}
        """
        # [MOCK DATA] - Estimate tokens (simple heuristic: ~4 chars per token)
        estimated_tokens = max(1, len(content) // 4)
        return {"total_tokens": estimated_tokens}


class MockGeminiConfig:
    """
    [MOCK API] - Mock Gemini configuration utility
    Simulates google.generativeai configuration
    """
    
    @staticmethod
    def configure(api_key: str):
        """
        [MOCK API] - Configure Gemini (no-op in mock)
        In production: genai.configure(api_key=api_key)
        """
        pass
    
    @staticmethod
    def GenerativeModel(model_name: str) -> MockGemini:
        """
        [MOCK API] - Create mock GenerativeModel
        In production: genai.GenerativeModel(model_name)
        """
        return MockGemini(api_key="mock_key_67890")
