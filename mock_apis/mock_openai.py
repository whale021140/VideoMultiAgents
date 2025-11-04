"""
[MOCK API] - Mock OpenAI API Implementation
This module simulates OpenAI API responses for demonstration purposes.
Replace with real openai.OpenAI() client in production.

Usage in production:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
"""

import json
from typing import List, Dict, Any


class MockMessage:
    """[MOCK API] - Represents a mock API response message"""
    def __init__(self, content: str):
        self.content = content


class MockChoice:
    """[MOCK API] - Represents a mock API response choice"""
    def __init__(self, message: MockMessage):
        self.message = message


class MockCompletion:
    """[MOCK API] - Represents a mock API completion response"""
    def __init__(self, content: str):
        self.choices = [MockChoice(MockMessage(content))]


class MockOpenAI:
    """
    [MOCK API] - Mock OpenAI client for demonstration
    Returns predefined responses for testing without API keys
    
    Replacement instructions for production:
    1. Replace with: from openai import OpenAI
    2. Initialize with: client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    3. Ensure environment variable OPENAI_API_KEY is set
    """
    
    def __init__(self, api_key: str = "mock_key_12345"):
        """
        [MOCK API] Initialize mock OpenAI client
        In production: OpenAI(api_key="your_real_key")
        """
        self.api_key = api_key
        self.model = "gpt-4o"
        
    class Chat:
        """[MOCK API] - Mock chat completion interface"""
        def __init__(self, parent):
            self.parent = parent
            
        class Completions:
            """[MOCK API] - Mock completions endpoint"""
            def __init__(self, parent):
                self.parent = parent
                
            def create(self, model: str = None, messages: List[Dict] = None, **kwargs) -> MockCompletion:
                """
                [MOCK API] - Mock chat completion create method
                
                In production:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content
                """
                # Extract the question from messages for better mock responses
                question_text = ""
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        question_text = msg.get("content", "")
                        break
                
                # [MOCK DATA] - Predefined responses based on input patterns
                mock_responses = {
                    "video": "This video demonstrates a common scenario involving visual analysis. The main subject appears to be engaged in typical activities. Key observations include: 1) Clear visual elements, 2) Logical sequence of events, 3) Identifiable patterns. Based on visual analysis, the predicted answer is: [DEMO RESPONSE]",
                    "question": "The answer to this question is determined through careful analysis of available information. Considering multiple perspectives and evidence, the most likely response is: [DEMO RESPONSE]",
                    "reasoning": "Analyzing the provided context: - First, we observe key elements - Then, we identify relationships - Finally, we conclude: [DEMO RESPONSE]",
                    "default": "Analysis complete. Response generated using mock API for demonstration purposes. Replace with real API for production use."
                }
                
                # Select appropriate mock response
                selected_response = mock_responses["default"]
                if "video" in question_text.lower():
                    selected_response = mock_responses["video"]
                elif "question" in question_text.lower():
                    selected_response = mock_responses["question"]
                elif "reason" in question_text.lower():
                    selected_response = mock_responses["reasoning"]
                
                # [MOCK API] Return mock completion
                return MockCompletion(selected_response)
        
        def __init__(self, parent):
            """[MOCK API] Initialize completions"""
            self.completions = self.Completions(parent)
    
    def __init__(self, api_key: str = "mock_key_12345"):
        """[MOCK API] Initialize mock OpenAI with chat interface"""
        self.api_key = api_key
        self.chat = self.Chat(self)
