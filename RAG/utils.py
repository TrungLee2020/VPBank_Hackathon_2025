
import re
import unicodedata
import hashlib
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

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

def estimate_tokens(text: str) -> int:
    """Estimate token count for Vietnamese text"""
    # Vietnamese specific estimation (roughly 1.2 tokens per word)
    words = len(text.split())
    return int(words * 1.2)