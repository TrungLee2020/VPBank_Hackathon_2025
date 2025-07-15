# --- config.py ---


import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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
    model_name: str = "thenlper/gte-small"
    model_dimension: int = 384
    batch_size: int = 32
    max_length: int = 512

@dataclass
class TableConfig:
    max_table_tokens: int = 1024
    table_summary_tokens: int = 256
    preserve_structure: bool = True
    extract_headers: bool = True

@dataclass
class BedrockConfig:
    aws_region: str = "us-east-1"
    model_id: str = "amazon.titan-text-express-v1"

@dataclass
class OpenSearchConfig:
    """Cấu hình để kết nối tới AWS OpenSearch Service."""
    # Lấy từ trang quản lý domain OpenSearch trên AWS console
    # Ví dụ: 'https://your-domain-name.us-east-1.es.amazonaws.com'
    hosts: List[str] = field(default_factory=lambda: ["https://localhost:9200"])
    
    # Sử dụng Basic Auth nếu bạn đã cấu hình Fine-Grained Access Control
    # Ví dụ: ('your_master_user', 'your_master_password')
    http_auth: Optional[Tuple[str, str]] = None 
    
    # Tên của index sẽ được tạo trong OpenSearch để lưu trữ vectors
    index_name: str = "vietnamese_rag_index"
    
    use_ssl: bool = True
    verify_certs: bool = True
    ssl_assert_hostname: bool = True
    ssl_show_warn: bool = True

@dataclass
class RerankerConfig:
    """Cấu hình cho bước reranking."""
    # Bật/tắt tính năng rerank
    enabled: bool = True 
    
    # Các lựa chọn khác: 'ms-marco-MiniLM-L-12-v2', 'cross-encoder/ms-marco-MiniLM-L-6-v2' cho local
    model_name: str = "BAAI/bge-reranker-v2-m3"
    
    # Model AWS
    modelId = "cohere.rerank-v3-5:0"
    model_package_arn = f"arn:aws:bedrock:{region}::foundation-model/{modelId}"

    # Số lượng kết quả sẽ giữ lại sau khi rerank
    top_n: int = 5
    
    # Kích thước batch khi rerank
    batch_size: int = 8

@dataclass
class RAGConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    table: TableConfig = field(default_factory=TableConfig)
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    opensearch: OpenSearchConfig = field(default_factory=OpenSearchConfig)
    
    # Thêm 'opensearch' vào danh sách các vector store được hỗ trợ
    vector_store_type: str = "opensearch"  # Có thể là "faiss" hoặc "opensearch"
    
    cache_enabled: bool = True
    cache_ttl: int = 3600