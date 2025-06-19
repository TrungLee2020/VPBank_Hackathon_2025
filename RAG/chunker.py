
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import math
from config import ChunkingConfig
from utils import estimate_tokens, generate_chunk_id, VietnameseTextProcessor
from preprocessor import DocumentStructure

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