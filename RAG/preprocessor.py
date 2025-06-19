
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from utils import VietnameseTextProcessor

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