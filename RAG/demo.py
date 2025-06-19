# Vietnamese RAG System - Core Modules
# File structure:
# vietnamese_rag/
# ├── __init__.py
# ├── config.py
# ├── preprocessor.py
# ├── chunker.py
# ├── embedder.py
# ├── retriever.py
# ├── table_processor.py
# └── utils.py

# =============================================================================
# config.py - Configuration Management
# =============================================================================

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class ChunkingConfig:
    parent_chunk_size: int = 2048
    child_chunk_size: int = 512
    overlap_tokens: int = 128
    min_chunk_size: int = 100
    max_chunk_size: int = 4096
    
@dataclass
class EmbeddingConfig:
    model_name: str = "dangvantuan/vietnamese-document-embedding"
    model_dimension: int = 768
    batch_size: int = 32
    max_length: int = 512

@dataclass
class TableConfig:
    max_table_tokens: int = 1024
    table_summary_tokens: int = 256
    preserve_structure: bool = True
    extract_headers: bool = True

@dataclass
class RAGConfig:
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    table: TableConfig = TableConfig()
    vector_store_type: str = "milvus"  # or "qdrant", "chroma"
    cache_enabled: bool = True
    cache_ttl: int = 3600

# Global config instance
config = RAGConfig()

# =============================================================================
# utils.py - Utility Functions
# =============================================================================

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

# =============================================================================
# preprocessor.py - Document Preprocessing
# =============================================================================

import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class DocumentStructure:
    title: str
    sections: List[Dict]
    tables: List[Dict]
    references: List[str]
    metadata: Dict

class VietnameseMarkdownPreprocessor:
    """Preprocess Vietnamese markdown documents"""
    
    def __init__(self):
        self.text_processor = VietnameseTextProcessor()
        self.md_parser = markdown.Markdown(
            extensions=['tables', 'toc', 'codehilite'],
            extension_configs={
                'toc': {'title': 'Mục lục'}
            }
        )
    
    def preprocess_document(self, content: str) -> DocumentStructure:
        """Main preprocessing pipeline"""
        # Normalize text
        normalized_content = self.text_processor.normalize_diacritics(content)
        
        # Extract document structure
        structure = self._extract_document_structure(normalized_content)
        
        # Extract tables
        tables = self._extract_tables(normalized_content)
        
        # Extract references
        references = self._extract_references(normalized_content)
        
        # Extract metadata
        metadata = self._extract_metadata(normalized_content)
        
        return DocumentStructure(
            title=metadata.get('title', 'Untitled'),
            sections=structure,
            tables=tables,
            references=references,
            metadata=metadata
        )
    
    def _extract_document_structure(self, content: str) -> List[Dict]:
        """Extract hierarchical structure from document"""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect headers
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    current_section['content'] = '\n'.join(current_content)
                    current_section['end_line'] = i - 1
                    sections.append(current_section)
                
                # Start new section
                header_level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('#').strip()
                
                current_section = {
                    'level': header_level,
                    'title': header_text,
                    'start_line': i,
                    'administrative_info': self._extract_administrative_info(header_text)
                }
                current_content = []
            
            # Detect administrative structures
            elif re.match(r'(?:PHẦN|CHƯƠNG|ĐIỀU)', line, re.IGNORECASE):
                if current_section:
                    current_section['content'] = '\n'.join(current_content)
                    current_section['end_line'] = i - 1
                    sections.append(current_section)
                
                current_section = {
                    'level': self._determine_admin_level(line),
                    'title': line,
                    'start_line': i,
                    'administrative_info': self._extract_administrative_info(line),
                    'type': 'administrative'
                }
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section:
            current_section['content'] = '\n'.join(current_content)
            current_section['end_line'] = len(lines) - 1
            sections.append(current_section)
        
        return sections
    
    def _extract_tables(self, content: str) -> List[Dict]:
        """Extract and parse markdown tables"""
        tables = []
        table_pattern = r'\|.*?\|'
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if re.match(table_pattern, line):
                # Found start of table
                table_lines = []
                table_start = i
                
                # Collect all table lines
                while i < len(lines) and re.match(table_pattern, lines[i].strip()):
                    table_lines.append(lines[i].strip())
                    i += 1
                
                if len(table_lines) >= 2:  # At least header + separator
                    table_data = self._parse_table_lines(table_lines)
                    tables.append({
                        'start_line': table_start,
                        'end_line': i - 1,
                        'raw_content': '\n'.join(table_lines),
                        'parsed_data': table_data,
                        'row_count': len(table_data.get('rows', [])),
                        'column_count': len(table_data.get('headers', []))
                    })
            else:
                i += 1
        
        return tables
    
    def _parse_table_lines(self, lines: List[str]) -> Dict:
        """Parse markdown table lines into structured data"""
        if len(lines) < 2:
            return {}
        
        # Parse headers
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
        
        # Skip separator line (line[1])
        # Parse data rows
        rows = []
        for line in lines[2:]:
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(cells) == len(headers):
                    rows.append(cells)
        
        return {
            'headers': headers,
            'rows': rows
        }
    
    def _extract_references(self, content: str) -> List[str]:
        """Extract internal and external references"""
        references = []
        
        # Vietnamese document reference patterns
        patterns = [
            r'(?:Điều|điều)\s+(\d+)',  # Article references
            r'(?:Khoản|khoản)\s+(\d+)',  # Clause references  
            r'(?:Chương|chương)\s+([IVX]+|\d+)',  # Chapter references
            r'(?:Phần|phần)\s+([IVX]+|\d+)',  # Section references
            r'(?:Quyết định|QĐ)\s+số\s+([\d\/\-A-Z]+)',  # Decision references
            r'(?:Thông tư|TT)\s+số\s+([\d\/\-A-Z]+)',  # Circular references
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))  # Remove duplicates
    
    def _extract_metadata(self, content: str) -> Dict:
        """Extract document metadata"""
        metadata = {}
        
        # Extract title (first header or document title pattern)
        title_match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Extract document number
        doc_number_pattern = r'Số:\s*([\d\/\-A-Z]+)'
        doc_number = re.search(doc_number_pattern, content)
        if doc_number:
            metadata['document_number'] = doc_number.group(1)
        
        # Extract date
        date_pattern = r'(?:ngày|Ngày)\s+(\d{1,2})\s+(?:tháng|th)\s+(\d{1,2})\s+(?:năm|nm)\s+(\d{4})'
        date_match = re.search(date_pattern, content)
        if date_match:
            day, month, year = date_match.groups()
            metadata['date'] = f"{day}/{month}/{year}"
        
        # Extract issuing authority
        authority_patterns = [
            r'(TỔNG CÔNG TY [^\\n]+)',
            r'(BỘ [^\\n]+)',
            r'(UBND [^\\n]+)'
        ]
        
        for pattern in authority_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata['issuing_authority'] = match.group(1).strip()
                break
        
        return metadata
    
    def _extract_administrative_info(self, text: str) -> Dict:
        """Extract administrative structure information"""
        info = {}
        
        # Section info
        section_match = re.search(r'(?:PHẦN|Phần)\s+([IVX]+|[0-9]+)', text, re.IGNORECASE)
        if section_match:
            info['section'] = section_match.group(1)
        
        # Chapter info  
        chapter_match = re.search(r'(?:CHƯƠNG|Chương)\s+([0-9]+)', text, re.IGNORECASE)
        if chapter_match:
            info['chapter'] = chapter_match.group(1)
        
        # Article info
        article_match = re.search(r'(?:ĐIỀU|Điều)\s+([0-9]+)', text, re.IGNORECASE)
        if article_match:
            info['article'] = article_match.group(1)
        
        return info
    
    def _determine_admin_level(self, text: str) -> int:
        """Determine administrative hierarchy level"""
        if re.search(r'(?:PHẦN|Phần)', text, re.IGNORECASE):
            return 1
        elif re.search(r'(?:CHƯƠNG|Chương)', text, re.IGNORECASE):
            return 2
        elif re.search(r'(?:ĐIỀU|Điều)', text, re.IGNORECASE):
            return 3
        elif re.search(r'^[0-9]+\.', text.strip()):
            return 4
        elif re.search(r'^[a-z]\)', text.strip()):
            return 5
        else:
            return 6

# =============================================================================
# chunker.py - Hierarchical Chunking
# =============================================================================

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class Chunk:
    id: str
    content: str
    chunk_type: str  # 'parent', 'child', 'table_parent', 'table_child'
    metadata: Dict
    parent_id: Optional[str] = None
    child_ids: List[str] = None
    tokens: int = 0
    
    def __post_init__(self):
        if self.child_ids is None:
            self.child_ids = []
        if self.tokens == 0:
            self.tokens = estimate_tokens(self.content)

class HierarchicalChunker:
    """Hierarchical chunking for Vietnamese documents"""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.text_processor = VietnameseTextProcessor()
    
    def chunk_document(self, doc_structure: DocumentStructure) -> List[Chunk]:
        """Main chunking pipeline"""
        all_chunks = []
        
        for section in doc_structure.sections:
            # Determine if section contains tables
            section_tables = self._get_section_tables(section, doc_structure.tables)
            
            if section_tables:
                chunks = self._chunk_section_with_tables(section, section_tables, doc_structure)
            else:
                chunks = self._chunk_text_section(section, doc_structure)
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _chunk_section_with_tables(self, section: Dict, tables: List[Dict], 
                                  doc_structure: DocumentStructure) -> List[Chunk]:
        """Chunk section containing tables"""
        chunks = []
        
        # Create table-aware parent chunk
        parent_content = self._create_table_aware_parent_content(section, tables)
        parent_metadata = self._create_parent_metadata(section, doc_structure, has_tables=True)
        
        parent_chunk = Chunk(
            id=generate_chunk_id(parent_content, parent_metadata),
            content=parent_content,
            chunk_type='table_parent',
            metadata=parent_metadata
        )
        
        # Create child chunks
        child_chunks = self._create_table_aware_children(section, tables, parent_chunk.id, doc_structure)
        
        # Update parent with child IDs
        parent_chunk.child_ids = [child.id for child in child_chunks]
        
        chunks.append(parent_chunk)
        chunks.extend(child_chunks)
        
        return chunks
    
    def _chunk_text_section(self, section: Dict, doc_structure: DocumentStructure) -> List[Chunk]:
        """Chunk text-only section"""
        chunks = []
        
        # Create parent chunk
        parent_content = self._create_contextual_content(section, doc_structure)
        parent_metadata = self._create_parent_metadata(section, doc_structure, has_tables=False)
        
        parent_chunk = Chunk(
            id=generate_chunk_id(parent_content, parent_metadata),
            content=parent_content,
            chunk_type='parent',
            metadata=parent_metadata
        )
        
        # Create child chunks
        child_chunks = self._create_text_children(section, parent_chunk.id, doc_structure)
        
        # Update parent with child IDs
        parent_chunk.child_ids = [child.id for child in child_chunks]
        
        chunks.append(parent_chunk)
        chunks.extend(child_chunks)
        
        return chunks
    
    def _create_table_aware_parent_content(self, section: Dict, tables: List[Dict]) -> str:
        """Create parent content that includes table context"""
        content_parts = []
        
        # Add section title and context
        if section.get('title'):
            content_parts.append(f"# {section['title']}")
        
        section_content = section.get('content', '')
        
        # Process content with table insertions
        lines = section_content.split('\n')
        current_content = []
        
        for i, line in enumerate(lines):
            # Check if this line starts a table
            table_at_line = self._find_table_at_line(i + section.get('start_line', 0), tables)
            
            if table_at_line:
                # Add accumulated content
                if current_content:
                    content_parts.append('\n'.join(current_content))
                    current_content = []
                
                # Add table with enhanced context
                table_context = self._create_table_context(table_at_line, section)
                content_parts.append(table_context)
                
                # Skip table lines
                table_line_count = table_at_line['end_line'] - table_at_line['start_line'] + 1
                for _ in range(table_line_count):
                    if i < len(lines):
                        i += 1
            else:
                current_content.append(line)
        
        # Add remaining content
        if current_content:
            content_parts.append('\n'.join(current_content))
        
        return '\n\n'.join(content_parts)
    
    def _create_table_aware_children(self, section: Dict, tables: List[Dict], 
                                   parent_id: str, doc_structure: DocumentStructure) -> List[Chunk]:
        """Create child chunks for section with tables"""
        children = []
        
        # Text-only children
        text_content = self._extract_text_without_tables(section, tables)
        if text_content.strip():
            text_children = self._split_text_into_chunks(
                text_content, 
                self.config.child_chunk_size,
                self.config.overlap_tokens
            )
            
            for i, chunk_text in enumerate(text_children):
                child_metadata = self._create_child_metadata(section, doc_structure, i, 'text')
                child = Chunk(
                    id=generate_chunk_id(chunk_text, child_metadata),
                    content=chunk_text,
                    chunk_type='child',
                    metadata=child_metadata,
                    parent_id=parent_id
                )
                children.append(child)
        
        # Table-specific children
        for table in tables:
            table_children = self._create_table_children(table, parent_id, section, doc_structure)
            children.extend(table_children)
        
        return children
    
    def _create_text_children(self, section: Dict, parent_id: str, 
                            doc_structure: DocumentStructure) -> List[Chunk]:
        """Create child chunks for text-only section"""
        children = []
        
        content = section.get('content', '')
        if not content.strip():
            return children
        
        # Split into child-sized chunks
        text_chunks = self._split_text_into_chunks(
            content,
            self.config.child_chunk_size, 
            self.config.overlap_tokens
        )
        
        for i, chunk_text in enumerate(text_chunks):
            # Add contextual information
            contextual_content = self._add_child_context(chunk_text, section, doc_structure)
            
            child_metadata = self._create_child_metadata(section, doc_structure, i, 'text')
            child = Chunk(
                id=generate_chunk_id(contextual_content, child_metadata),
                content=contextual_content,
                chunk_type='child',
                metadata=child_metadata,
                parent_id=parent_id
            )
            children.append(child)
        
        return children
    
    def _split_text_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        # Estimate current tokens
        current_tokens = estimate_tokens(text)
        
        if current_tokens <= chunk_size:
            return [text]
        
        # Split by sentences to respect boundaries
        sentences = self.text_processor.extract_vietnamese_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk, overlap)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(estimate_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """Get sentences for overlap based on token count"""
        if not sentences:
            return []
        
        overlap_sentences = []
        total_tokens = 0
        
        # Work backwards from end of chunk
        for sentence in reversed(sentences):
            sentence_tokens = estimate_tokens(sentence)
            if total_tokens + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                total_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _create_parent_metadata(self, section: Dict, doc_structure: DocumentStructure, 
                              has_tables: bool) -> Dict:
        """Create metadata for parent chunk"""
        metadata = {
            'chunk_level': 'parent',
            'section_title': section.get('title', ''),
            'section_level': section.get('level', 0),
            'document_title': doc_structure.title,
            'has_tables': has_tables,
            'administrative_info': section.get('administrative_info', {}),
            'start_line': section.get('start_line', 0),
            'end_line': section.get('end_line', 0)
        }
        
        # Add hierarchy path
        hierarchy_path = self._build_hierarchy_path(section, doc_structure)
        metadata['hierarchy_path'] = hierarchy_path
        
        return metadata
    
    def _create_child_metadata(self, section: Dict, doc_structure: DocumentStructure,
                             chunk_index: int, content_type: str) -> Dict:
        """Create metadata for child chunk"""
        metadata = {
            'chunk_level': 'child',
            'chunk_index': chunk_index,
            'content_type': content_type,  # 'text', 'table_summary', 'table_data'
            'section_title': section.get('title', ''),
            'section_level': section.get('level', 0),
            'document_title': doc_structure.title,
            'administrative_info': section.get('administrative_info', {})
        }
        
        return metadata
    
    def _build_hierarchy_path(self, section: Dict, doc_structure: DocumentStructure) -> str:
        """Build hierarchy path string"""
        admin_info = section.get('administrative_info', {})
        path_parts = []
        
        if 'section' in admin_info:
            path_parts.append(f"Phần {admin_info['section']}")
        if 'chapter' in admin_info:
            path_parts.append(f"Chương {admin_info['chapter']}")
        if 'article' in admin_info:
            path_parts.append(f"Điều {admin_info['article']}")
        
        if not path_parts and section.get('title'):
            path_parts.append(section['title'])
        
        return ' > '.join(path_parts)
    
    def _get_section_tables(self, section: Dict, all_tables: List[Dict]) -> List[Dict]:
        """Get tables that belong to this section"""
        section_start = section.get('start_line', 0)
        section_end = section.get('end_line', float('inf'))
        
        section_tables = []
        for table in all_tables:
            table_start = table.get('start_line', 0)
            if section_start <= table_start <= section_end:
                section_tables.append(table)
        
        return section_tables
    
    def _find_table_at_line(self, line_num: int, tables: List[Dict]) -> Optional[Dict]:
        """Find table that starts at given line"""
        for table in tables:
            if table.get('start_line', 0) == line_num:
                return table
        return None
    
    def _create_table_context(self, table: Dict, section: Dict) -> str:
        """Create enhanced context for table"""
        context_parts = []
        
        # Add table summary
        parsed_data = table.get('parsed_data', {})
        headers = parsed_data.get('headers', [])
        row_count = table.get('row_count', 0)
        
        summary = f"**Bảng ({row_count} hàng, {len(headers)} cột)**"
        if headers:
            summary += f" - Các cột: {', '.join(headers)}"
        
        context_parts.append(summary)
        context_parts.append(table.get('raw_content', ''))
        
        return '\n'.join(context_parts)
    
    def _extract_text_without_tables(self, section: Dict, tables: List[Dict]) -> str:
        """Extract text content excluding tables"""
        content = section.get('content', '')
        lines = content.split('\n')
        
        # Remove table lines
        filtered_lines = []
        section_start = section.get('start_line', 0)
        
        for i, line in enumerate(lines):
            line_num = section_start + i
            
            # Check if this line is part of a table
            is_table_line = False
            for table in tables:
                if table.get('start_line', 0) <= line_num <= table.get('end_line', 0):
                    is_table_line = True
                    break
            
            if not is_table_line:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _create_table_children(self, table: Dict, parent_id: str, section: Dict,
                             doc_structure: DocumentStructure) -> List[Chunk]:
        """Create child chunks for table"""
        children = []
        
        parsed_data = table.get('parsed_data', {})
        headers = parsed_data.get('headers', [])
        rows = parsed_data.get('rows', [])
        
        # Table summary child
        summary_content = self._create_table_summary(table, section)
        summary_metadata = self._create_child_metadata(section, doc_structure, 0, 'table_summary')
        summary_metadata['table_info'] = {
            'headers': headers,
            'row_count': len(rows),
            'column_count': len(headers)
        }
        
        summary_child = Chunk(
            id=generate_chunk_id(summary_content, summary_metadata),
            content=summary_content,
            chunk_type='table_child',
            metadata=summary_metadata,
            parent_id=parent_id
        )
        children.append(summary_child)
        
        # Table data child(s)
        if len(rows) <= 10:  # Small table - single chunk
            data_content = self._format_table_data(headers, rows)
            data_metadata = self._create_child_metadata(section, doc_structure, 1, 'table_data')
            data_metadata['table_info'] = summary_metadata['table_info']
            
            data_child = Chunk(
                id=generate_chunk_id(data_content, data_metadata),
                content=data_content,
                chunk_type='table_child',
                metadata=data_metadata,
                parent_id=parent_id
            )
            children.append(data_child)
        
        else:  # Large table - split into multiple chunks
            chunk_size = 5  # rows per chunk
            for i in range(0, len(rows), chunk_size):
                chunk_rows = rows[i:i + chunk_size]
                data_content = self._format_table_data(headers, chunk_rows)
                data_metadata = self._create_child_metadata(section, doc_structure, i + 1, 'table_data')
                data_metadata['table_info'] = {
                    'headers': headers,
                    'row_count': len(chunk_rows),
                    'column_count': len(headers),
                    'chunk_start_row': i,
                    'chunk_end_row': i + len(chunk_rows) - 1
                }
                
                data_child = Chunk(
                    id=generate_chunk_id(data_content, data_metadata),
                    content=data_content,
                    chunk_type='table_child',
                    metadata=data_metadata,
                    parent_id=parent_id
                )
                children.append(data_child)
        
        return children
    
    def _create_table_summary(self, table: Dict, section: Dict) -> str:
        """Create summary description of table"""
        parsed_data = table.get('parsed_data', {})
        headers = parsed_data.get('headers', [])
        rows = parsed_data.get('rows', [])
        
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Bảng trong {section.get('title', 'phần này')} có {len(rows)} hàng và {len(headers)} cột.")
        
        # Column info
        if headers:
            summary_parts.append(f"Các cột bao gồm: {', '.join(headers)}")
        
        # Sample data
        if rows:
            summary_parts.append("Dữ liệu mẫu:")
            for i, row in enumerate(rows[:3]):  # First 3 rows
                row_data = []
                for j, cell in enumerate(row):
                    if j < len(headers):
                        row_data.append(f"{headers[j]}: {cell}")
                summary_parts.append(f"- Hàng {i+1}: {'; '.join(row_data)}")
            
            if len(rows) > 3:
                summary_parts.append(f"... và {len(rows) - 3} hàng khác")
        
        return '\n'.join(summary_parts)
    
    def _format_table_data(self, headers: List[str], rows: List[List[str]]) -> str:
        """Format table data as text"""
        if not headers or not rows:
            return ""
        
        # Create formatted table
        lines = []
        
        # Header
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join([" --- " for _ in headers]) + "|")
        
        # Rows
        for row in rows:
            # Pad row to match header count
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded_row[:len(headers)]) + " |")
        
        return '\n'.join(lines)
    
    def _add_child_context(self, content: str, section: Dict, doc_structure: DocumentStructure) -> str:
        """Add contextual information to child chunk"""
        context_parts = []
        
        # Document context
        context_parts.append(f"Tài liệu: {doc_structure.title}")
        
        # Section context
        if section.get('title'):
            context_parts.append(f"Phần: {section['title']}")
        
        # Administrative context
        admin_info = section.get('administrative_info', {})
        if admin_info:
            admin_context = []
            if 'section' in admin_info:
                admin_context.append(f"Phần {admin_info['section']}")
            if 'chapter' in admin_info:
                admin_context.append(f"Chương {admin_info['chapter']}")
            if 'article' in admin_info:
                admin_context.append(f"Điều {admin_info['article']}")
            
            if admin_context:
                context_parts.append(" > ".join(admin_context))
        
        # Combine with content
        context_prefix = "**Bối cảnh:** " + " | ".join(context_parts) + "\n\n"
        return context_prefix + content
    
    def _create_contextual_content(self, section: Dict, doc_structure: DocumentStructure) -> str:
        """Create contextual content for parent chunk"""
        content_parts = []
        
        # Add hierarchical context
        hierarchy_path = self._build_hierarchy_path(section, doc_structure)
        if hierarchy_path:
            content_parts.append(f"**Vị trí trong tài liệu:** {hierarchy_path}")
        
        # Add section title
        if section.get('title'):
            content_parts.append(f"# {section['title']}")
        
        # Add main content
        content_parts.append(section.get('content', ''))
        
        return '\n\n'.join(content_parts)

# =============================================================================
# table_processor.py - Advanced Table Processing
# =============================================================================

import pandas as pd
from typing import List, Dict, Optional, Tuple
import re

class TableProcessor:
    """Advanced table processing for Vietnamese documents"""
    
    def __init__(self, config: TableConfig = None):
        self.config = config or TableConfig()
    
    def process_table(self, table_data: Dict) -> Dict:
        """Process table with Vietnamese-specific enhancements"""
        processed_data = table_data.copy()
        
        # Extract and clean data
        headers = table_data.get('parsed_data', {}).get('headers', [])
        rows = table_data.get('parsed_data', {}).get('rows', [])
        
        if not headers or not rows:
            return processed_data
        
        # Clean headers
        clean_headers = self._clean_vietnamese_headers(headers)
        
        # Clean cell data
        clean_rows = self._clean_table_rows(rows)
        
        # Detect data types
        column_types = self._detect_column_types(clean_rows, clean_headers)
        
        # Create enhanced summary
        enhanced_summary = self._create_enhanced_table_summary(
            clean_headers, clean_rows, column_types
        )
        
        # Update processed data
        processed_data['processed_data'] = {
            'headers': clean_headers,
            'rows': clean_rows,
            'column_types': column_types,
            'enhanced_summary': enhanced_summary
        }
        
        return processed_data
    
    def _clean_vietnamese_headers(self, headers: List[str]) -> List[str]:
        """Clean and normalize Vietnamese table headers"""
        clean_headers = []
        
        for header in headers:
            # Remove extra whitespace
            clean_header = re.sub(r'\s+', ' ', header.strip())
            
            # Normalize common Vietnamese abbreviations
            abbreviations = {
                'STT': 'Số thứ tự',
                'TT': 'Thứ tự', 
                'SL': 'Số lượng',
                'DT': 'Doanh thu',
                'ĐVHC': 'Đơn vị hành chính',
                'TCTD': 'Tổ chức tín dụng'
            }
            
            for abbr, full in abbreviations.items():
                if clean_header.upper() == abbr:
                    clean_header = f"{full} ({abbr})"
                    break
            
            clean_headers.append(clean_header)
        
        return clean_headers
    
    def _clean_table_rows(self, rows: List[List[str]]) -> List[List[str]]:
        """Clean table row data"""
        clean_rows = []
        
        for row in rows:
            clean_row = []
            for cell in row:
                # Clean cell content
                clean_cell = re.sub(r'\s+', ' ', str(cell).strip())
                
                # Handle common Vietnamese formatting
                # Remove thousand separators in numbers
                if re.match(r'^[\d\.,]+, clean_cell):
                    clean_cell = clean_cell.replace(',', '')
                
                clean_row.append(clean_cell)
            
            clean_rows.append(clean_row)
        
        return clean_rows
    
    def _detect_column_types(self, rows: List[List[str]], headers: List[str]) -> Dict[str, str]:
        """Detect data types for each column"""
        column_types = {}
        
        if not rows:
            return column_types
        
        for col_idx, header in enumerate(headers):
            column_values = []
            
            # Collect values for this column
            for row in rows:
                if col_idx < len(row):
                    column_values.append(row[col_idx])
            
            # Detect type
            column_type = self._infer_column_type(column_values, header)
            column_types[header] = column_type
        
        return column_types
    
    def _infer_column_type(self, values: List[str], header: str) -> str:
        """Infer column data type"""
        if not values:
            return 'text'
        
        # Check for Vietnamese-specific patterns
        header_lower = header.lower()
        
        # Date columns
        if any(keyword in header_lower for keyword in ['ngày', 'tháng', 'năm', 'thời gian']):
            return 'date'
        
        # Currency columns
        if any(keyword in header_lower for keyword in ['tiền', 'đồng', 'giá', 'phí', 'lương']):
            return 'currency'
        
        # Number columns
        if any(keyword in header_lower for keyword in ['số', 'lượng', 'tỷ lệ', '%']):
            # Check if values are numeric
            numeric_count = 0
            for value in values[:10]:  # Sample first 10 values
                if re.match(r'^[\d\.,\s]+, value.strip()):
                    numeric_count += 1
            
            if numeric_count > len(values[:10]) * 0.7:  # 70% numeric
                return 'number'
        
        # Organization/person names
        if any(keyword in header_lower for keyword in ['tên', 'họ', 'cơ quan', 'đơn vị', 'công ty']):
            return 'name'
        
        # Administrative codes
        if any(keyword in header_lower for keyword in ['mã', 'số hiệu', 'quyết định']):
            return 'code'
        
        return 'text'
    
    def _create_enhanced_table_summary(self, headers: List[str], rows: List[List[str]], 
                                     column_types: Dict[str, str]) -> str:
        """Create enhanced summary with Vietnamese context"""
        summary_parts = []
        
        # Basic statistics
        summary_parts.append(f"Bảng dữ liệu có {len(rows)} hàng và {len(headers)} cột")
        
        # Column descriptions
        summary_parts.append("\n**Mô tả các cột:**")
        for header in headers:
            col_type = column_types.get(header, 'text')
            type_description = self._get_type_description(col_type)
            summary_parts.append(f"- {header}: {type_description}")
        
        # Data insights
        insights = self._generate_table_insights(headers, rows, column_types)
        if insights:
            summary_parts.append(f"\n**Thông tin quan trọng:**")
            summary_parts.extend([f"- {insight}" for insight in insights])
        
        return '\n'.join(summary_parts)
    
    def _get_type_description(self, col_type: str) -> str:
        """Get Vietnamese description for column type"""
        descriptions = {
            'text': 'Dữ liệu văn bản',
            'number': 'Dữ liệu số',
            'currency': 'Dữ liệu tiền tệ',
            'date': 'Dữ liệu ngày tháng',
            'name': 'Tên người/tổ chức',
            'code': 'Mã số/ký hiệu'
        }
        return descriptions.get(col_type, 'Dữ liệu văn bản')
    
    def _generate_table_insights(self, headers: List[str], rows: List[List[str]], 
                               column_types: Dict[str, str]) -> List[str]:
        """Generate insights about table data"""
        insights = []
        
        # Find key columns
        key_columns = []
        for header in headers:
            header_lower = header.lower()
            if any(keyword in header_lower for keyword in ['tên', 'số', 'mã', 'loại']):
                key_columns.append(header)
        
        if key_columns:
            insights.append(f"Các cột quan trọng: {', '.join(key_columns)}")
        
        # Numeric insights
        for header, col_type in column_types.items():
            if col_type in ['number', 'currency']:
                col_idx = headers.index(header)
                values = []
                
                for row in rows:
                    if col_idx < len(row):
                        try:
                            # Try to convert to number
                            val_str = row[col_idx].replace(',', '').replace('.', '')
                            if val_str.isdigit():
                                values.append(int(val_str))
                        except:
                            continue
                
                if values:
                    insights.append(f"{header}: từ {min(values):,} đến {max(values):,}")
        
        return insights

# =============================================================================
# embedder.py - Vietnamese Embedding
# =============================================================================

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor

class VietnameseEmbedder:
    """Vietnamese-optimized embedding for RAG"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()
    
    def _load_model(self):
        """Load Vietnamese embedding model"""
        try:
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.device
            )
            logger.info(f"Loaded model {self.config.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to multilingual model
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            logger.info("Using fallback multilingual model")
    
    def embed_chunks(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        """Embed list of chunks"""
        embeddings = {}
        
        # Group chunks by type for optimized processing
        text_chunks = [c for c in chunks if c.chunk_type in ['parent', 'child']]
        table_chunks = [c for c in chunks if 'table' in c.chunk_type]
        
        # Process text chunks
        if text_chunks:
            text_embeddings = self._embed_text_chunks(text_chunks)
            embeddings.update(text_embeddings)
        
        # Process table chunks
        if table_chunks:
            table_embeddings = self._embed_table_chunks(table_chunks)
            embeddings.update(table_embeddings)
        
        return embeddings
    
    def _embed_text_chunks(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        """Embed text chunks with batching"""
        embeddings = {}
        
        # Prepare texts
        texts = []
        chunk_ids = []
        
        for chunk in chunks:
            # Preprocess text for embedding
            processed_text = self._preprocess_for_embedding(chunk.content)
            texts.append(processed_text)
            chunk_ids.append(chunk.id)
        
        # Batch embedding
        try:
            batch_embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Map back to chunk IDs
            for chunk_id, embedding in zip(chunk_ids, batch_embeddings):
                embeddings[chunk_id] = embedding
                
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            # Fallback to individual embedding
            for chunk_id, text in zip(chunk_ids, texts):
                try:
                    embedding = self.model.encode([text])[0]
                    embeddings[chunk_id] = embedding
                except Exception as e2:
                    logger.error(f"Failed to embed chunk {chunk_id}: {e2}")
        
        return embeddings
    
    def _embed_table_chunks(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        """Embed table chunks with special handling"""
        embeddings = {}
        
        for chunk in chunks:
            try:
                if chunk.chunk_type == 'table_child':
                    # Special processing for table content
                    processed_text = self._preprocess_table_for_embedding(chunk)
                else:
                    processed_text = self._preprocess_for_embedding(chunk.content)
                
                embedding = self.model.encode([processed_text])[0]
                embeddings[chunk.id] = embedding
                
            except Exception as e:
                logger.error(f"Failed to embed table chunk {chunk.id}: {e}")
        
        return embeddings
    
    def _preprocess_for_embedding(self, text: str) -> str:
        """Preprocess Vietnamese text for embedding"""
        # Normalize
        text = VietnameseTextProcessor.normalize_diacritics(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic
        text = re.sub(r'#{1,6}\s+', '', text)         # Remove headers
        
        # Truncate if too long
        words = text.split()
        if len(words) > self.config.max_length:
            text = ' '.join(words[:self.config.max_length])
        
        return text.strip()
    
    def _preprocess_table_for_embedding(self, chunk: Chunk) -> str:
        """Special preprocessing for table chunks"""
        content = chunk.content
        
        # Extract table info from metadata
        table_info = chunk.metadata.get('table_info', {})
        
        # Create structured representation
        if table_info:
            structured_text = []
            
            # Add table description
            headers = table_info.get('headers', [])
            if headers:
                structured_text.append(f"Bảng với các cột: {', '.join(headers)}")
            
            # Add row count info
            row_count = table_info.get('row_count', 0)
            if row_count:
                structured_text.append(f"Có {row_count} hàng dữ liệu")
            
            # Add actual content
            structured_text.append(content)
            
            return ' | '.join(structured_text)
        
        return self._preprocess_for_embedding(content)
    
    async def embed_chunks_async(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        """Async embedding for large batches"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Split chunks into batches
            batch_size = 100
            chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            
            # Process batches in parallel
            tasks = []
            for batch in chunk_batches:
                task = loop.run_in_executor(executor, self.embed_chunks, batch)
                tasks.append(task)
            
            # Combine results
            batch_results = await asyncio.gather(*tasks)
            
            combined_embeddings = {}
            for result in batch_results:
                combined_embeddings.update(result)
            
            return combined_embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed search query"""
        processed_query = self._preprocess_for_embedding(query)
        return self.model.encode([processed_query])[0]

# =============================================================================
# retriever.py - Hierarchical Retrieval
# =============================================================================

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import faiss
from abc import ABC, abstractmethod

@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    chunk: Chunk
    parent_chunk: Optional[Chunk] = None

class VectorStore(ABC):
    """Abstract vector store interface"""
    
    @abstractmethod
    def add_embeddings(self, embeddings: Dict[str, np.ndarray], chunks: List[Chunk]):
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int, filter_dict: Dict = None) -> List[RetrievalResult]:
        pass
    
    @abstractmethod
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        pass

class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.chunk_map = {}  # chunk_id -> Chunk
        self.id_map = {}     # faiss_id -> chunk_id
        self.embedding_map = {}  # chunk_id -> embedding
        self.next_id = 0
    
    def add_embeddings(self, embeddings: Dict[str, np.ndarray], chunks: List[Chunk]):
        """Add embeddings to FAISS index"""
        chunk_dict = {chunk.id: chunk for chunk in chunks}
        
        embeddings_list = []
        chunk_ids = []
        
        for chunk_id, embedding in embeddings.items():
            if chunk_id in chunk_dict:
                # Normalize embedding for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    normalized_embedding = embedding / norm
                    embeddings_list.append(normalized_embedding)
                    chunk_ids.append(chunk_id)
                    
                    # Update mappings
                    self.chunk_map[chunk_id] = chunk_dict[chunk_id]
                    self.id_map[self.next_id] = chunk_id
                    self.embedding_map[chunk_id] = normalized_embedding
                    self.next_id += 1
        
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype('float32')
            self.index.add(embeddings_array)
            logger.info(f"Added {len(embeddings_list)} embeddings to FAISS index")
    
    def search(self, query_embedding: np.ndarray, k: int, filter_dict: Dict = None) -> List[RetrievalResult]:
        """Search for similar chunks"""
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # Search in FAISS
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            min(k * 2, self.index.ntotal)  # Get more candidates for filtering
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            chunk_id = self.id_map.get(idx)
            if not chunk_id:
                continue
                
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            
            # Apply filters
            if filter_dict and not self._matches_filter(chunk, filter_dict):
                continue
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                score=float(score),
                chunk=chunk
            )
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID"""
        return self.chunk_map.get(chunk_id)
    
    def _matches_filter(self, chunk: Chunk, filter_dict: Dict) -> bool:
        """Check if chunk matches filter criteria"""
        for key, value in filter_dict.items():
            if key == 'chunk_type':
                if isinstance(value, list):
                    if chunk.chunk_type not in value:
                        return False
                elif chunk.chunk_type != value:
                    return False
            elif key in chunk.metadata:
                if chunk.metadata[key] != value:
                    return False
        return True

class HierarchicalRetriever:
    """Hierarchical retrieval system"""
    
    def __init__(self, vector_store: VectorStore, embedder: VietnameseEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, k: int = 5, retrieval_strategy: str = 'hierarchical') -> List[RetrievalResult]:
        """Main retrieval function"""
        query_embedding = self.embedder.embed_query(query)
        
        if retrieval_strategy == 'hierarchical':
            return self._hierarchical_retrieve(query_embedding, k)
        elif retrieval_strategy == 'table_aware':
            return self._table_aware_retrieve(query, query_embedding, k)
        else:
            return self._simple_retrieve(query_embedding, k)
    
    def _hierarchical_retrieve(self, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
        """Hierarchical retrieval: child chunks -> parent chunks with enhanced context"""
        # Step 1: Retrieve child chunks for precision
        child_results = self.vector_store.search(
            query_embedding,
            k=k * 3,  # Get more child candidates
            filter_dict={'chunk_type': ['child', 'table_child']}
        )
        
        # Step 2: Get parent chunks for context and ensure we have complete information
        final_results = []
        seen_parents = set()
        
        for child_result in child_results[:k * 2]:  # Consider more candidates
            chunk = child_result.chunk
            
            # Always try to get parent chunk for broader context
            parent_chunk = None
            if chunk.parent_id:
                parent_chunk = self.vector_store.get_chunk_by_id(chunk.parent_id)
            
            # Create enhanced result with guaranteed parent context
            enhanced_result = RetrievalResult(
                chunk_id=chunk.id,
                score=child_result.score,
                chunk=chunk,
                parent_chunk=parent_chunk
            )
            
            # Add to results - we want both precise child and broad parent context
            final_results.append(enhanced_result)
            
            # Also add standalone parent if it's significantly different and relevant
            if parent_chunk and parent_chunk.id not in seen_parents:
                seen_parents.add(parent_chunk.id)
                
                # Check if parent chunk would be relevant on its own
                parent_score = self._calculate_parent_relevance(parent_chunk, query_embedding)
                if parent_score > 0.3:  # Threshold for parent relevance
                    parent_result = RetrievalResult(
                        chunk_id=parent_chunk.id,
                        score=parent_score,
                        chunk=parent_chunk,
                        parent_chunk=None  # Parent chunk as standalone
                    )
                    final_results.append(parent_result)
            
            if len(final_results) >= k:
                break
        
        # Sort by score and return top k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:k]
    
    def _calculate_parent_relevance(self, parent_chunk: Chunk, query_embedding: np.ndarray) -> float:
        """Calculate relevance score for parent chunk"""
        try:
            # Get parent embedding if available
            parent_embedding = self.vector_store.embedding_map.get(parent_chunk.id)
            if parent_embedding is not None:
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, parent_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(parent_embedding)
                )
                return float(similarity)
        except:
            pass
        
        return 0.0  # Default low score if can't calculate
    
    def _table_aware_retrieve(self, query: str, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
        """Table-aware retrieval"""
        # Classify query type
        query_type = self._classify_query_type(query)
        
        if query_type == 'table_query':
            # Prioritize table chunks
            table_results = self.vector_store.search(
                query_embedding,
                k=k,
                filter_dict={'chunk_type': ['table_child']}
            )
            
            # Enhance with parent context
            enhanced_results = []
            for result in table_results:
                if result.chunk.parent_id:
                    parent_chunk = self.vector_store.get_chunk_by_id(result.chunk.parent_id)
                    result.parent_chunk = parent_chunk
                enhanced_results.append(result)
            
            return enhanced_results
        else:
            # Standard hierarchical retrieval
            return self._hierarchical_retrieve(query_embedding, k)
    
    def _simple_retrieve(self, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
        """Simple similarity search"""
        return self.vector_store.search(query_embedding, k)
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for appropriate retrieval strategy"""
        query_lower = query.lower()
        
        # Table query indicators
        table_keywords = [
            'bảng', 'biểu', 'danh sách', 'thống kê', 'số liệu',
            'cột', 'hàng', 'dữ liệu', 'tỷ lệ', 'phần trăm'
        ]
        
        if any(keyword in query_lower for keyword in table_keywords):
            return 'table_query'
        
        return 'text_query'
    
    def format_context(self, results: List[RetrievalResult], max_tokens: int = 4096) -> Dict[str, str]:
        """Format retrieval results into comprehensive context for LLM"""
        formatted_contexts = []
        current_tokens = 0
        
        for i, result in enumerate(results):
            # Build comprehensive context with both child and parent
            context_item = self._build_comprehensive_context(result, i + 1)
            
            # Estimate tokens for this context item
            item_tokens = estimate_tokens(context_item['combined_content'])
            
            # Check if we have space
            if current_tokens + item_tokens > max_tokens:
                # Try to fit with truncation
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 200:  # Only add if meaningful space left
                    truncated_context = self._truncate_context(context_item, remaining_tokens)
                    formatted_contexts.append(truncated_context)
                    current_tokens = max_tokens
                break
            else:
                formatted_contexts.append(context_item)
                current_tokens += item_tokens
            
            if current_tokens >= max_tokens:
                break
        
        # Combine all contexts
        final_context = self._combine_contexts(formatted_contexts)
        
        return {
            'context': final_context,
            'context_breakdown': formatted_contexts,
            'total_tokens': current_tokens
        }
    
    def _build_comprehensive_context(self, result: RetrievalResult, source_number: int) -> Dict[str, str]:
        """Build comprehensive context with both child precision and parent breadth"""
        child_chunk = result.chunk
        parent_chunk = result.parent_chunk
        
        # Build context header with metadata
        metadata = child_chunk.metadata
        context_header = f"**Nguồn {source_number}**"
        
        if metadata.get('hierarchy_path'):
            context_header += f" ({metadata['hierarchy_path']})"
        
        context_header += f" - Độ liên quan: {result.score:.3f}"
        
        # Combine child and parent content strategically
        if parent_chunk and parent_chunk.id != child_chunk.id:
            # We have both child and parent - provide full context
            
            # Get the precise match (child) content
            child_content = child_chunk.content
            parent_content = parent_chunk.content
            
            # Create layered context
            if child_chunk.chunk_type == 'table_child':
                # For table child, prioritize table context but include section context
                combined_content = f"""
{context_header}

**Nội dung chính (từ bảng):**
{child_content}

**Bối cảnh đầy đủ (từ phần cha):**
{parent_content}
"""
            else:
                # For text child, show precise match + broader context
                combined_content = f"""
{context_header}

**Thông tin cụ thể (phần được tìm thấy):**
{child_content}

**Bối cảnh rộng hơn (toàn bộ phần):**
{parent_content}
"""
        else:
            # Only have one chunk (probably parent or standalone)
            chunk_to_use = parent_chunk if parent_chunk else child_chunk
            combined_content = f"""
{context_header}

**Nội dung đầy đủ:**
{chunk_to_use.content}
"""
        
        return {
            'source_number': source_number,
            'header': context_header,
            'child_content': child_chunk.content,
            'parent_content': parent_chunk.content if parent_chunk else None,
            'combined_content': combined_content,
            'metadata': metadata,
            'chunk_type': child_chunk.chunk_type
        }
    
    def _truncate_context(self, context_item: Dict, max_tokens: int) -> Dict:
        """Truncate context while preserving important information"""
        # Priority: header + child_content + portion of parent_content
        header = context_item['header']
        child_content = context_item['child_content']
        parent_content = context_item['parent_content']
        
        header_tokens = estimate_tokens(header)
        child_tokens = estimate_tokens(child_content)
        
        available_for_parent = max_tokens - header_tokens - child_tokens - 50  # Buffer
        
        if parent_content and available_for_parent > 100:
            # Truncate parent content to fit
            parent_words = parent_content.split()
            target_words = int(available_for_parent * 0.8)  # Conservative estimate
            
            if len(parent_words) > target_words:
                truncated_parent = ' '.join(parent_words[:target_words]) + "\n\n[...nội dung đã được rút gọn...]"
            else:
                truncated_parent = parent_content
            
            combined_content = f"""
{header}

**Thông tin cụ thể:**
{child_content}

**Bối cảnh rộng hơn:**
{truncated_parent}
"""
        else:
            # Only include child content
            combined_content = f"""
{header}

**Nội dung:**
{child_content}
"""
        
        context_item_copy = context_item.copy()
        context_item_copy['combined_content'] = combined_content
        context_item_copy['truncated'] = True
        
        return context_item_copy
    
    def _combine_contexts(self, formatted_contexts: List[Dict]) -> str:
        """Combine all formatted contexts into final context string"""
        if not formatted_contexts:
            return "Không tìm thấy thông tin liên quan."
        
        context_parts = []
        
        # Add summary header
        context_parts.append(f"**Tìm thấy {len(formatted_contexts)} nguồn thông tin liên quan:**\n")
        
        # Add each context
        for context_item in formatted_contexts:
            context_parts.append(context_item['combined_content'])
            context_parts.append("=" * 80)  # Separator
        
        return "\n\n".join(context_parts)

# =============================================================================
# Main RAG System Class
# =============================================================================

class VietnameseRAGSystem:
    """Main RAG system orchestrating all components"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # Initialize components
        self.preprocessor = VietnameseMarkdownPreprocessor()
        self.chunker = HierarchicalChunker(self.config.chunking)
        self.table_processor = TableProcessor(self.config.table)
        self.embedder = VietnameseEmbedder(self.config.embedding)
        
        # Initialize vector store
        self.vector_store = self._create_vector_store()
        self.retriever = HierarchicalRetriever(self.vector_store, self.embedder)
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0
        }
    
    def _create_vector_store(self) -> VectorStore:
        """Create vector store based on configuration"""
        if self.config.vector_store_type == 'faiss':
            return FAISSVectorStore(dimension=self.config.embedding.model_dimension)
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.vector_store_type}")
    
    async def process_document(self, content: str, document_id: str = None) -> Dict:
        """Process a single document through the full pipeline"""
        try:
            # Step 1: Preprocess document
            logger.info(f"Preprocessing document {document_id}")
            doc_structure = self.preprocessor.preprocess_document(content)
            
            # Step 2: Process tables
            logger.info("Processing tables")
            for i, table in enumerate(doc_structure.tables):
                doc_structure.tables[i] = self.table_processor.process_table(table)
            
            # Step 3: Create chunks
            logger.info("Creating hierarchical chunks")
            chunks = self.chunker.chunk_document(doc_structure)
            
            # Step 4: Generate embeddings
            logger.info("Generating embeddings")
            embeddings = await self.embedder.embed_chunks_async(chunks)
            
            # Step 5: Store in vector database
            logger.info("Storing in vector database")
            self.vector_store.add_embeddings(embeddings, chunks)
            
            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            self.stats['embeddings_generated'] += len(embeddings)
            
            return {
                'status': 'success',
                'document_id': document_id,
                'chunks_created': len(chunks),
                'embeddings_generated': len(embeddings),
                'tables_processed': len(doc_structure.tables)
            }
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            return {
                'status': 'error',
                'document_id': document_id,
                'error': str(e)
            }
    
    async def process_documents_batch(self, documents: List[Tuple[str, str]]) -> List[Dict]:
        """Process multiple documents in batch"""
        tasks = []
        for content, doc_id in documents:
            task = self.process_document(content, doc_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    def query(self, question: str, k: int = 5, strategy: str = 'hierarchical') -> Dict:
        """Query the RAG system with enhanced parent context"""
        try:
            # Retrieve relevant chunks
            results = self.retriever.retrieve(
                query=question,
                k=k,
                retrieval_strategy=strategy
            )
            
            if not results:
                return {
                    'answer': 'Không tìm thấy thông tin liên quan đến câu hỏi.',
                    'context': '',
                    'context_breakdown': [],
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Format context with comprehensive parent information
            context_result = self.retriever.format_context(results)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(results)
            
            # Prepare detailed sources with parent context info
            sources = []
            for result in results:
                source_info = {
                    'chunk_id': result.chunk_id,
                    'score': result.score,
                    'title': result.chunk.metadata.get('section_title', 'Unknown'),
                    'hierarchy_path': result.chunk.metadata.get('hierarchy_path', ''),
                    'chunk_type': result.chunk.chunk_type,
                    'has_parent_context': result.parent_chunk is not None,
                    'parent_chunk_id': result.parent_chunk.id if result.parent_chunk else None
                }
                sources.append(source_info)
            
            return {
                'context': context_result['context'],
                'context_breakdown': context_result['context_breakdown'],
                'total_context_tokens': context_result['total_tokens'],
                'sources': sources,
                'confidence': confidence,
                'strategy_used': strategy,
                'results_count': len(results)
            }
            
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return {
                'answer': f'Lỗi khi xử lý câu hỏi: {str(e)}',
                'context': '',
                'context_breakdown': [],
                'sources': [],
                'confidence': 0.0
            }
    
    def _calculate_confidence(self, results: List[RetrievalResult]) -> float:
        """Calculate confidence score based on retrieval results"""
        if not results:
            return 0.0
        
        # Average of top scores with weighted decay
        weights = [0.4, 0.3, 0.2, 0.1]  # Weight for top 4 results
        weighted_score = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(results[:4]):
            weight = weights[i] if i < len(weights) else 0.05
            weighted_score += result.score * weight
            total_weight += weight
        
        return min(weighted_score / total_weight if total_weight > 0 else 0.0, 1.0)
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return self.stats.copy()
    
    def save_index(self, filepath: str):
        """Save vector index to disk"""
        if hasattr(self.vector_store, 'index'):
            faiss.write_index(self.vector_store.index, f"{filepath}.faiss")
            
            # Save metadata
            import pickle
            metadata = {
                'chunk_map': self.vector_store.chunk_map,
                'id_map': self.vector_store.id_map,
                'embedding_map': self.vector_store.embedding_map,
                'next_id': self.vector_store.next_id,
                'stats': self.stats
            }
            
            with open(f"{filepath}.metadata", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load vector index from disk"""
        try:
            # Load FAISS index
            self.vector_store.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            import pickle
            with open(f"{filepath}.metadata", 'rb') as f:
                metadata = pickle.load(f)
            
            self.vector_store.chunk_map = metadata['chunk_map']
            self.vector_store.id_map = metadata['id_map']
            self.vector_store.embedding_map = metadata['embedding_map']
            self.vector_store.next_id = metadata['next_id']
            self.stats = metadata.get('stats', self.stats)
            
            logger.info(f"Index loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")

# =============================================================================
# Usage Example
# =============================================================================

async def example_usage():
    """Example of how to use the Vietnamese RAG system"""
    
    # Initialize system
    config = RAGConfig()
    rag_system = VietnameseRAGSystem(config)
    
    # Sample Vietnamese administrative document
    sample_document = """
# TỔNG CÔNG TY BƯU ĐIỆN VIỆT NAM

## PHẦN I: QUY ĐỊNH CHUNG

### Điều 1. Phạm vi điều chỉnh
Văn bản này quy định về tổ chức và hoạt động của các đơn vị thuộc Tổng công ty.

### Điều 2. Đối tượng áp dụng
| STT | Đối tượng | Phạm vi áp dụng |
|-----|-----------|-----------------|
| 1 | Bưu điện tỉnh | Toàn quốc |
| 2 | Trung tâm kỹ thuật | Toàn quốc |
| 3 | Công ty con | Theo quy định |

## PHẦN II: TỔ CHỨC THỰC HIỆN

### Điều 3. Trách nhiệm các đơn vị
1. Bưu điện tỉnh chịu trách nhiệm triển khai
2. Trung tâm kỹ thuật hỗ trợ kỹ thuật
"""
    
    # Process document
    result = await rag_system.process_document(sample_document, "sample_doc_1")
    print(f"Processing result: {result}")
    
    # Query the system
    query_result = rag_system.query(
        "Trách nhiệm của Bưu điện tỉnh là gì?",
        k=3,
        strategy='hierarchical'
    )
    
    print(f"Query result: {query_result}")
    
    # Get statistics
    stats = rag_system.get_statistics()
    print(f"System stats: {stats}")

# Run example
if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())