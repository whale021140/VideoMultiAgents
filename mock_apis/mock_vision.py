"""
[MOCK API] - Mock Vision Feature Extraction
This module provides mock implementations of video/image feature extraction.
Replace with real vision models (CLIP, ViT, etc.) in production.

Usage in production:
    from transformers import CLIPProcessor, CLIPModel
    # or use other vision models for feature extraction
"""

import json
import numpy as np
from typing import Dict, List, Any, Union, Tuple


class MockVisionExtractor:
    """
    [MOCK API] - Mock Vision Feature Extractor
    Returns predefined feature vectors and extracted information for demonstration
    
    Replacement instructions for production:
    1. Replace with real vision models (CLIP, ViT, ResNet, etc.)
    2. Load model: model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    3. Process images and get real embeddings
    4. Ensure GPU/CUDA availability for production
    """
    
    def __init__(self, model_name: str = "mock-vision-v1", embedding_dim: int = 768):
        """
        [MOCK API] Initialize mock vision extractor
        
        Args:
            model_name: [MOCK] Model identifier (unused in mock)
            embedding_dim: [MOCK] Dimension of mock embeddings
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
    
    def extract_frame_features(self, frame_data: Union[str, np.ndarray]) -> np.ndarray:
        """
        [MOCK API] - Extract features from a single frame
        
        In production:
            image = Image.open(frame_path)
            processed = processor(images=image, return_tensors="pt")
            embeddings = model.get_image_features(**processed)
            return embeddings.detach().cpu().numpy()
        
        Args:
            frame_data: Path to image or image array
        
        Returns:
            [MOCK DATA] - Random feature vector of shape (embedding_dim,)
        """
        # [MOCK DATA] Generate pseudo-random but deterministic features
        np.random.seed(hash(str(frame_data)) % 2**32)
        features = np.random.randn(self.embedding_dim).astype(np.float32)
        # Normalize features
        features = features / (np.linalg.norm(features) + 1e-8)
        return features
    
    def extract_video_features(self, video_path: str, num_frames: int = 8) -> np.ndarray:
        """
        [MOCK API] - Extract features from video frames
        
        In production:
            # Use video frame extraction + real vision model
            frames = extract_frames_from_video(video_path, num_frames)
            features = [extract_frame_features(f) for f in frames]
            return np.stack(features)
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
        
        Returns:
            [MOCK DATA] - Feature matrix of shape (num_frames, embedding_dim)
        """
        # [MOCK DATA] Generate features for sampled frames
        features = []
        for i in range(num_frames):
            # Generate deterministic features based on frame index and video path
            np.random.seed((hash(video_path) + i) % 2**32)
            frame_features = np.random.randn(self.embedding_dim).astype(np.float32)
            frame_features = frame_features / (np.linalg.norm(frame_features) + 1e-8)
            features.append(frame_features)
        
        return np.stack(features)
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """
        [MOCK API] - Extract semantic features from text
        
        In production:
            # Use text encoder from CLIP or similar
            encoded = processor(text=[text], return_tensors="pt")
            embeddings = model.get_text_features(**encoded)
            return embeddings.detach().cpu().numpy()
        
        Args:
            text: Input text string
        
        Returns:
            [MOCK DATA] - Text embedding vector of shape (embedding_dim,)
        """
        # [MOCK DATA] Generate deterministic text features
        np.random.seed(hash(text) % 2**32)
        features = np.random.randn(self.embedding_dim).astype(np.float32)
        features = features / (np.linalg.norm(features) + 1e-8)
        return features
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        [MOCK API] - Compute cosine similarity between feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
        
        Returns:
            Cosine similarity score in range [-1, 1]
        """
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def extract_scene_information(self, frame_data: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        [MOCK API] - Extract scene information from frame
        
        In production:
            # Use object detection, scene understanding models
            # e.g., DETR for object detection, scene classification models
        
        Returns:
            [MOCK DATA] - Dictionary with scene information
        """
        # [MOCK DATA] Predefined scene information
        mock_scene_info = {
            "objects": ["person", "table", "chair", "computer"],
            "scene_type": "office",
            "dominant_color": "white",
            "lighting": "indoor",
            "activity": "work",
            "confidence_scores": {
                "scene_classification": 0.85,
                "object_detection": 0.78,
                "activity_recognition": 0.72
            }
        }
        
        return mock_scene_info


class MockVideoProcessor:
    """[MOCK API] - Utility class for video processing in mock mode"""
    
    @staticmethod
    def get_frame_count(video_path: str) -> int:
        """[MOCK API] - Get frame count from video"""
        # [MOCK DATA] Return fixed frame count for demo
        return 240  # Represents a 10-second video at 24fps
    
    @staticmethod
    def sample_frames(video_path: str, num_samples: int = 8) -> List[str]:
        """
        [MOCK API] - Sample frames from video
        
        Returns:
            [MOCK DATA] - List of frame identifiers (in mock, just strings)
        """
        return [f"frame_{i}" for i in range(num_samples)]
