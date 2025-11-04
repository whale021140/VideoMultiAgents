"""
Comment-to-QA Converter: Two-stage Gemini-powered processor

Stage 1: Quality Assessment - Identify comments suitable for VideoQA
Stage 2: Question Generation - Create 5-choice MCQ from high-quality comments

This module extracts video QA questions from user comments using Gemini API.
"""

import os
import json
import re
from typing import List, Dict, Optional
from util import ask_gpt4_omni


# Stage 1: Quality Assessment Prompt
STAGE1_PROMPT_TEMPLATE = """
You are an expert at identifying comments that can be transformed into video question-answering (QA) tasks.

Analyze the following comment and determine if it is suitable for creating a multiple-choice question about video content.

A suitable comment should:
1. Reference specific video content or actions
2. Ask or describe something observable in the video
3. Be substantive (not just emoji or generic praise)
4. Be transformable into a clear, answerable question

Comment: {comment_text}

Respond with ONLY "yes" or "no".
"""

# Stage 2: Question Generation Prompt
STAGE2_PROMPT_TEMPLATE = """
You are an expert video content analyst. Create a multiple-choice question based on the following comment about a video.

The question should:
1. Be clear and specific about what is being asked
2. Reference the video content implied by the comment
3. Have exactly 5 distinct, plausible options (A, B, C, D, E)
4. Be answerable from video observation

Comment (timestamp {timestamp}): {comment_text}

Generate a JSON response with this exact structure:
{{
    "question": "Clear, specific question about the video",
    "option_a": "First option",
    "option_b": "Second option",
    "option_c": "Third option",
    "option_d": "Fourth option",
    "option_e": "Fifth option"
}}

Respond with ONLY the JSON, no additional text.
"""


class CommentQAExtractor:
    """
    Two-stage Gemini-powered comment-to-question converter.
    
    Stage 1: Quality Assessment - Identify comments suitable for QA
    Stage 2: Question Generation - Create 5-choice MCQ from comment
    """
    
    def __init__(self, gemini_api_key: str):
        """
        Initialize CommentQAExtractor with Gemini API credentials.
        
        Args:
            gemini_api_key: Google Gemini API key
        
        Raises:
            ValueError: If API key is empty
        """
        if not gemini_api_key:
            raise ValueError("Gemini API key cannot be empty")
        self.gemini_api_key = gemini_api_key
        self.processed_count = 0
        self.passed_count = 0
        self.failed_count = 0
    
    def extract_questions_from_comments(
        self, 
        video_id: str, 
        comments_list: List[Dict]
    ) -> Dict:
        """
        Main entry point: Convert comments to QA dataset.
        
        Executes two-stage processing:
        - Stage 1: Filter comments by quality/suitability
        - Stage 2: Generate questions and options for suitable comments
        
        Args:
            video_id: Video identifier
            comments_list: List of comment dicts with keys:
                - 'text': comment text (required)
                - 'timestamp': video timestamp (required)
                - 'comment_id': unique identifier (optional)
        
        Returns:
            Dict with structure:
            {
                "video_id": str,
                "questions": [
                    {
                        "q_uid": str,
                        "question": str,
                        "option 0": str,
                        "option 1": str,
                        "option 2": str,
                        "option 3": str,
                        "option 4": str,
                        "source_comment": str,
                        "source_comment_id": str,
                        "timestamp": str
                    },
                    ...
                ]
            }
        """
        questions = []
        
        print(f"\n[CommentQAExtractor] Processing video: {video_id}")
        print(f"[Stage 1] Assessing comment quality...")
        
        # Stage 1: Filter comments by quality
        suitable_comments = []
        for comment in comments_list:
            comment_text = comment.get("text", "").strip()
            comment_id = comment.get("comment_id", "unknown")
            
            if not comment_text:
                print(f"  [Stage 1] {comment_id}: SKIP (empty text)")
                continue
            
            # Truncate display text
            display_text = comment_text[:60] + "..." if len(comment_text) > 60 else comment_text
            
            is_suitable = self._assess_comment_quality(comment_text)
            
            if is_suitable:
                suitable_comments.append(comment)
                print(f"  [Stage 1] {comment_id}: PASS → {display_text}")
            else:
                print(f"  [Stage 1] {comment_id}: SKIP → {display_text}")
        
        self.processed_count = len(comments_list)
        self.passed_count = len(suitable_comments)
        
        print(f"\n[Stage 2] Generating QA from {len(suitable_comments)} suitable comments...")
        
        # Stage 2: Generate questions from suitable comments
        for comment in suitable_comments:
            comment_text = comment.get("text", "")
            timestamp = comment.get("timestamp", "00:00")
            comment_id = comment.get("comment_id", "unknown")
            
            qa_dict = self._generate_qa_from_comment(comment_text, timestamp)
            
            if qa_dict:
                # Construct full QA entry
                q_uid = f"{video_id}_{comment_id}"
                qa_entry = {
                    "q_uid": q_uid,
                    "question": qa_dict.get("question", ""),
                    "option 0": qa_dict.get("option 0", ""),
                    "option 1": qa_dict.get("option 1", ""),
                    "option 2": qa_dict.get("option 2", ""),
                    "option 3": qa_dict.get("option 3", ""),
                    "option 4": qa_dict.get("option 4", ""),
                    "source_comment": comment_text,
                    "source_comment_id": comment_id,
                    "timestamp": timestamp
                }
                questions.append(qa_entry)
                print(f"  [Stage 2] {comment_id}: SUCCESS")
            else:
                self.failed_count += 1
                print(f"  [Stage 2] {comment_id}: FAILED")
        
        result = {
            "video_id": video_id,
            "questions": questions
        }
        
        print(f"\n[Summary] {len(questions)}/{self.processed_count} comments converted to QA")
        
        return result
    
    def _assess_comment_quality(self, comment_text: str) -> bool:
        """
        Stage 1: Use Gemini to assess if comment is suitable for QA extraction.
        
        Determines if a comment contains enough substantive information about
        video content to be transformed into a multiple-choice question.
        
        Args:
            comment_text: Raw comment text
        
        Returns:
            True if suitable for question generation, False otherwise
        """
        try:
            prompt = STAGE1_PROMPT_TEMPLATE.format(comment_text=comment_text)
            response = ask_gpt4_omni(
                gemini_api_key=self.gemini_api_key,
                prompt_text=prompt,
                temperature=0.1  # Low temperature for consistent yes/no
            )
            return "yes" in response.lower()
        except Exception as e:
            print(f"    [Error in Stage 1] {str(e)}")
            return False
    
    def _generate_qa_from_comment(
        self, 
        comment_text: str, 
        timestamp: str
    ) -> Optional[Dict]:
        """
        Stage 2: Use Gemini to generate question and 5 options from comment.
        
        Creates a complete multiple-choice question with 5 plausible options
        based on the substantive content of the comment.
        
        Args:
            comment_text: Comment that passed Stage 1 quality check
            timestamp: Video timestamp (e.g., "00:10")
        
        Returns:
            Dict with structure:
            {
                "question": str,
                "option 0": str,
                "option 1": str,
                "option 2": str,
                "option 3": str,
                "option 4": str
            }
            
            Returns None if generation fails or JSON parsing error occurs.
            
        Note: Keyframe retrieval is a placeholder for future implementation.
              Currently just stores timestamp as-is.
              TODO: Implement get_keyframes_by_timestamp(video_id, timestamp)
        """
        try:
            prompt = STAGE2_PROMPT_TEMPLATE.format(
                comment_text=comment_text,
                timestamp=timestamp
            )
            response = ask_gpt4_omni(
                gemini_api_key=self.gemini_api_key,
                prompt_text=prompt,
                temperature=0.3  # Moderate temperature for diverse but reasonable options
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                print(f"    [Error] No JSON found in response")
                return None
            
            qa_dict = json.loads(json_match.group())
            
            # Normalize option keys from option_a/b/c/d/e to option 0-4
            normalized = {
                "question": qa_dict.get("question", ""),
                "option 0": qa_dict.get("option_a", ""),
                "option 1": qa_dict.get("option_b", ""),
                "option 2": qa_dict.get("option_c", ""),
                "option 3": qa_dict.get("option_d", ""),
                "option 4": qa_dict.get("option_e", "")
            }
            
            # Verify all fields exist
            if not all(normalized.values()):
                print(f"    [Error] Missing required fields in generated QA")
                return None
            
            return normalized
            
        except json.JSONDecodeError as e:
            print(f"    [Error] JSON parsing failed: {str(e)}")
            return None
        except Exception as e:
            print(f"    [Error in Stage 2] {str(e)}")
            return None
    
    def get_stats(self) -> Dict:
        """
        Return processing statistics.
        
        Returns:
            Dict with keys:
            - processed_count: total comments processed
            - passed_count: comments that passed Stage 1
            - failed_count: comments that failed Stage 2
        """
        return {
            "processed_count": self.processed_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count
        }
