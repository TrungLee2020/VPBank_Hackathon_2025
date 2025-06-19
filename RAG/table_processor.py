
import pandas as pd
from typing import List, Dict, Optional, Tuple
import re
from config import TableConfig
import logging

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
                if re.match(r'^[\d\.,]+$', clean_cell):
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
                if re.match(r'^[\d\.,\s]+$', value.strip()):
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