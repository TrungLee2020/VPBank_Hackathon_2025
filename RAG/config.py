
import os
from dataclasses import dataclass, field
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
    model_name: str = "thenlper/gte-small" #"dangvantuan/vietnamese-document-embedding"
    model_dimension: int = 384 #768
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
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    table: TableConfig = field(default_factory=TableConfig)
    vector_store_type: str = "faiss"  # or "milvus"  # or "qdrant", "chroma"
    cache_enabled: bool = True
    cache_ttl: int = 3600


# Global config instance
# config = RAGConfig()