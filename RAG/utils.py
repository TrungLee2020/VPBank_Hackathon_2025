# --- utils.py ---


import re
import unicodedata
import hashlib
from typing import List, Dict, Tuple, Optional
import logging
import json
from abc import ABC, abstractmethod
from config import BedrockConfig

# Cần cài đặt boto3: pip install boto3
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
except ImportError:
    boto3 = None
    NoCredentialsError = None
    ClientError = None


logger = logging.getLogger(__name__)

# --- Tokenizer Abstraction ---

class Tokenizer(ABC):
    """Abstract base class for tokenizers."""
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a given text."""
        pass

class HeuristicTokenizer(Tokenizer):
    """
    A simple heuristic-based tokenizer for Vietnamese text.
    Estimates token count based on word count.
    """
    def count_tokens(self, text: str) -> int:
        """Estimate token count for Vietnamese text."""
        # Vietnamese specific estimation (roughly 1.2 tokens per word)
        words = len(text.split())
        return int(words * 1.2)

class BedrockTokenizer(Tokenizer):
    """
    A tokenizer that uses an AWS Bedrock model to get an accurate token count.
    """
    def __init__(self, config: BedrockConfig):
        if boto3 is None:
            raise ImportError("boto3 is not installed. Please install it using 'pip install boto3'")
        self.config = config
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime', 
                region_name=self.config.aws_region
            )
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure them (e.g., using 'aws configure').")
            self.bedrock_client = None

    def count_tokens(self, text: str) -> int:
        """
        Counts tokens by invoking a Bedrock model.
        Falls back to heuristic if the API call fails.
        """
        if not self.bedrock_client:
            logger.warning("Bedrock client not available. Falling back to heuristic token counting.")
            return HeuristicTokenizer().count_tokens(text)

        # Payload for Amazon Titan models
        body = json.dumps({
            "inputText": text,
            "textGenerationConfig": {
                "maxTokenCount": 1, # We don't need to generate, just count
                "temperature": 0,
                "stopSequences": []
            }
        })
        
        try:
            # We use invoke_model to get the token count from the output
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=self.config.model_id,
                accept='application/json',
                contentType='application/json'
            )
            response_body = json.loads(response.get('body').read())
            
            # The token count is typically in a field like 'inputTextTokenCount' for Titan
            if 'inputTextTokenCount' in response_body:
                return response_body['inputTextTokenCount']
            else:
                logger.warning(f"Could not find token count in Bedrock response for model {self.config.model_id}. Falling back to heuristic.")
                return HeuristicTokenizer().count_tokens(text)

        except ClientError as e:
            logger.error(f"An error occurred with Bedrock API: {e}. Falling back to heuristic.")
            return HeuristicTokenizer().count_tokens(text)
        except Exception as e:
            logger.error(f"An unexpected error occurred while counting tokens: {e}. Falling back to heuristic.")
            return HeuristicTokenizer().count_tokens(text)


class VietnameseTextProcessor:
    """Vietnamese text processing utilities"""
    
    @staticmethod
    def normalize_diacritics(text: str) -> str:
        """Normalize Vietnamese diacritics"""
        # Convert to NFC form (canonical decomposition followed by canonical composition)
        normalized = unicodedata.normalize('NFC', text)
        return normalized
    
    @staticmethod
    def extract_vietnamese_sentences(text: str) -> List[str]:
        """Split Vietnamese text into sentences"""
        # Vietnamese sentence boundaries
        sentence_endings = r'[.!?;]\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def detect_administrative_structure(text: str) -> Dict[str, List[str]]:
        """Detect Vietnamese administrative document structure"""
        structure = {
            'sections': [],  # Phần I, II, III
            'chapters': [],  # Chương 1, 2, 3  
            'articles': [],  # Điều 1, 2, 3
            'points': []     # 1., 2., a), b)
        }
        
        # Section pattern: Phần I, II, III or PHẦN I, II, III
        section_pattern = r'(?:PHẦN|Phần)\s+([IVX]+|[0-9]+)'
        structure['sections'] = re.findall(section_pattern, text, re.IGNORECASE)
        
        # Chapter pattern: Chương 1, 2, 3
        chapter_pattern = r'(?:CHƯƠNG|Chương)\s+([0-9]+)'
        structure['chapters'] = re.findall(chapter_pattern, text, re.IGNORECASE)
        
        # Article pattern: Điều 1, 2, 3
        article_pattern = r'(?:ĐIỀU|Điều)\s+([0-9]+)'
        structure['articles'] = re.findall(article_pattern, text, re.IGNORECASE)
        
        # Point pattern: 1., 2., a), b), c)
        point_pattern = r'^([0-9]+\.|[a-z]\))'
        for line in text.split('\n'):
            if re.match(point_pattern, line.strip()):
                structure['points'].append(line.strip())
        
        return structure

def generate_chunk_id(content: str, metadata: Dict = None) -> str:
    """Generate unique ID for chunk"""
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    if metadata:
        meta_str = str(sorted(metadata.items()))
        meta_hash = hashlib.md5(meta_str.encode('utf-8')).hexdigest()[:8]
        return f"{content_hash}_{meta_hash}"
    return content_hash

# Hàm estimate_tokens cũ được giữ lại để dùng trong HeuristicTokenizer
def estimate_tokens(text: str) -> int:
    """Estimate token count for Vietnamese text"""
    # Vietnamese specific estimation (roughly 1.2 tokens per word)
    words = len(text.split())
    return int(words * 1.2)