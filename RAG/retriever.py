
# from typing import List, Dict, Optional, Tuple
# import numpy as np
# from dataclasses import dataclass
# import faiss
# from abc import ABC, abstractmethod
# from chunker import Chunk
# from embedder import VietnameseEmbedder
# from utils import estimate_tokens
# import logging
# logger = logging.getLogger(__name__)

# @dataclass
# class RetrievalResult:
#     chunk_id: str
#     score: float
#     chunk: Chunk
#     parent_chunk: Optional[Chunk] = None

# class VectorStore(ABC):
#     """Abstract vector store interface"""
    
#     @abstractmethod
#     def add_embeddings(self, embeddings: Dict[str, np.ndarray], chunks: List[Chunk]):
#         pass
    
#     @abstractmethod
#     def search(self, query_embedding: np.ndarray, k: int, filter_dict: Dict = None) -> List[RetrievalResult]:
#         pass
    
#     @abstractmethod
#     def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
#         pass

# class FAISSVectorStore(VectorStore):
#     """FAISS-based vector store implementation"""
    
#     def __init__(self, dimension: int = 768):
#         self.dimension = dimension
#         self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
#         self.chunk_map = {}  # chunk_id -> Chunk
#         self.id_map = {}     # faiss_id -> chunk_id
#         self.embedding_map = {}  # chunk_id -> embedding
#         self.next_id = 0
    
#     def add_embeddings(self, embeddings: Dict[str, np.ndarray], chunks: List[Chunk]):
#         """Add embeddings to FAISS index"""
#         chunk_dict = {chunk.id: chunk for chunk in chunks}
        
#         embeddings_list = []
#         chunk_ids = []
        
#         for chunk_id, embedding in embeddings.items():
#             if chunk_id in chunk_dict:
#                 # Normalize embedding for cosine similarity
#                 norm = np.linalg.norm(embedding)
#                 if norm > 0:
#                     normalized_embedding = embedding / norm
#                     embeddings_list.append(normalized_embedding)
#                     chunk_ids.append(chunk_id)
                    
#                     # Update mappings
#                     self.chunk_map[chunk_id] = chunk_dict[chunk_id]
#                     self.id_map[self.next_id] = chunk_id
#                     self.embedding_map[chunk_id] = normalized_embedding
#                     self.next_id += 1
        
#         if embeddings_list:
#             embeddings_array = np.array(embeddings_list).astype('float32')
#             self.index.add(embeddings_array)
#             logger.info(f"Added {len(embeddings_list)} embeddings to FAISS index")
    
#     def search(self, query_embedding: np.ndarray, k: int, filter_dict: Dict = None) -> List[RetrievalResult]:
#         """Search for similar chunks"""
#         # Normalize query embedding
#         norm = np.linalg.norm(query_embedding)
#         if norm > 0:
#             query_embedding = query_embedding / norm
        
#         # Search in FAISS
#         scores, indices = self.index.search(
#             query_embedding.reshape(1, -1).astype('float32'), 
#             min(k * 2, self.index.ntotal)  # Get more candidates for filtering
#         )
        
#         results = []
#         for score, idx in zip(scores[0], indices[0]):
#             if idx == -1:  # Invalid index
#                 continue
                
#             chunk_id = self.id_map.get(idx)
#             if not chunk_id:
#                 continue
                
#             chunk = self.chunk_map.get(chunk_id)
#             if not chunk:
#                 continue
            
#             # Apply filters
#             if filter_dict and not self._matches_filter(chunk, filter_dict):
#                 continue
            
#             result = RetrievalResult(
#                 chunk_id=chunk_id,
#                 score=float(score),
#                 chunk=chunk
#             )
#             results.append(result)
            
#             if len(results) >= k:
#                 break
        
#         return results
    
#     def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
#         """Get chunk by ID"""
#         return self.chunk_map.get(chunk_id)
    
#     def _matches_filter(self, chunk: Chunk, filter_dict: Dict) -> bool:
#         """Check if chunk matches filter criteria"""
#         for key, value in filter_dict.items():
#             if key == 'chunk_type':
#                 if isinstance(value, list):
#                     if chunk.chunk_type not in value:
#                         return False
#                 elif chunk.chunk_type != value:
#                     return False
#             elif key in chunk.metadata:
#                 if chunk.metadata[key] != value:
#                     return False
#         return True

# class HierarchicalRetriever:
#     """Hierarchical retrieval system"""
    
#     def __init__(self, vector_store: VectorStore, embedder: VietnameseEmbedder):
#         self.vector_store = vector_store
#         self.embedder = embedder
    
#     def retrieve(self, query: str, k: int = 5, retrieval_strategy: str = 'hierarchical') -> List[RetrievalResult]:
#         """Main retrieval function"""
#         query_embedding = self.embedder.embed_query(query)
        
#         if retrieval_strategy == 'hierarchical':
#             return self._hierarchical_retrieve(query_embedding, k)
#         elif retrieval_strategy == 'table_aware':
#             return self._table_aware_retrieve(query, query_embedding, k)
#         else:
#             return self._simple_retrieve(query_embedding, k)
    
#     def _hierarchical_retrieve(self, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
#         """Hierarchical retrieval: child chunks -> parent chunks"""
#         # Step 1: Retrieve child chunks for precision
#         child_results = self.vector_store.search(
#             query_embedding,
#             k=k * 3,  # Get more child candidates
#             filter_dict={'chunk_type': ['child', 'table_child']}
#         )
        
#         # Step 2: Get parent chunks for context
#         final_results = []
#         seen_parents = set()
        
#         for child_result in child_results[:k]:
#             chunk = child_result.chunk
            
#             # Get parent chunk if available
#             parent_chunk = None
#             if chunk.parent_id:
#                 parent_chunk = self.vector_store.get_chunk_by_id(chunk.parent_id)
                
#                 # Avoid duplicate parents
#                 if parent_chunk and parent_chunk.id not in seen_parents:
#                     seen_parents.add(parent_chunk.id)
                    
#                     # Create result with parent context
#                     result = RetrievalResult(
#                         chunk_id=chunk.id,
#                         score=child_result.score,
#                         chunk=chunk,
#                         parent_chunk=parent_chunk
#                     )
#                     final_results.append(result)
#             else:
#                 # Standalone chunk
#                 final_results.append(child_result)
            
#             if len(final_results) >= k:
#                 break
        
#         return final_results
    
#     def _table_aware_retrieve(self, query: str, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
#         """Table-aware retrieval"""
#         # Classify query type
#         query_type = self._classify_query_type(query)
        
#         if query_type == 'table_query':
#             # Prioritize table chunks
#             table_results = self.vector_store.search(
#                 query_embedding,
#                 k=k,
#                 filter_dict={'chunk_type': ['table_child']}
#             )
            
#             # Enhance with parent context
#             enhanced_results = []
#             for result in table_results:
#                 if result.chunk.parent_id:
#                     parent_chunk = self.vector_store.get_chunk_by_id(result.chunk.parent_id)
#                     result.parent_chunk = parent_chunk
#                 enhanced_results.append(result)
            
#             return enhanced_results
#         else:
#             # Standard hierarchical retrieval
#             return self._hierarchical_retrieve(query_embedding, k)
    
#     def _simple_retrieve(self, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
#         """Simple similarity search"""
#         return self.vector_store.search(query_embedding, k)
    
#     def _classify_query_type(self, query: str) -> str:
#         """Classify query type for appropriate retrieval strategy"""
#         query_lower = query.lower()
        
#         # Table query indicators
#         table_keywords = [
#             'bảng', 'biểu', 'danh sách', 'thống kê', 'số liệu',
#             'cột', 'hàng', 'dữ liệu', 'tỷ lệ', 'phần trăm'
#         ]
        
#         if any(keyword in query_lower for keyword in table_keywords):
#             return 'table_query'
        
#         return 'text_query'
    
#     def format_context(self, results: List[RetrievalResult], max_tokens: int = 4096) -> str:
#         """Format retrieval results into context for LLM"""
#         context_parts = []
#         current_tokens = 0
        
#         for i, result in enumerate(results):
#             # Use parent chunk if available for better context
#             chunk_to_use = result.parent_chunk if result.parent_chunk else result.chunk
            
#             # Add metadata context
#             metadata = chunk_to_use.metadata
#             context_header = f"**Nguồn {i+1}**"
            
#             if metadata.get('hierarchy_path'):
#                 context_header += f" ({metadata['hierarchy_path']})"
            
#             context_header += f" - Độ liên quan: {result.score:.3f}"
            
#             # Add content
#             chunk_content = chunk_to_use.content
#             chunk_tokens = estimate_tokens(chunk_content)
            
#             # Check if we have space
#             if current_tokens + chunk_tokens > max_tokens:
#                 # Truncate content to fit
#                 remaining_tokens = max_tokens - current_tokens
#                 if remaining_tokens > 100:  # Only add if meaningful space left
#                     words = chunk_content.split()
#                     truncated_words = words[:int(remaining_tokens * 0.8)]  # Conservative estimate
#                     chunk_content = ' '.join(truncated_words) + "..."
#                     chunk_tokens = remaining_tokens
#                 else:
#                     break
            
#             context_parts.append(f"{context_header}\n{chunk_content}")
#             current_tokens += chunk_tokens + 20  # Header overhead
            
#             if current_tokens >= max_tokens:
#                 break
        
#         return "\n\n" + "="*50 + "\n\n".join(context_parts)

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import faiss
import pickle
import os
from abc import ABC, abstractmethod
from chunker import Chunk
from embedder import VietnameseEmbedder
from utils import estimate_tokens
import logging

logger = logging.getLogger(__name__)

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
    
    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> bool:
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> bool:
        pass

class FAISSVectorStore(VectorStore):
    """Enhanced FAISS-based vector store with HNSW support"""
    
    def __init__(self, 
                 dimension: int = 768,
                 index_type: str = 'hnsw',
                 m: int = 32,
                 ef_construction: int = 200,
                 ef_search: int = 128,
                 metric: str = 'cosine'):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Embedding dimension
            index_type: 'flat', 'hnsw', or 'ivf'
            m: Number of bi-directional links for HNSW
            ef_construction: Size of dynamic candidate list for HNSW construction
            ef_search: Size of dynamic candidate list for HNSW search
            metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Create index based on type
        self.index = self._create_index()
        
        # Mappings
        self.chunk_map = {}  # chunk_id -> Chunk
        self.id_map = {}     # faiss_id -> chunk_id
        self.reverse_id_map = {}  # chunk_id -> faiss_id
        self.embedding_map = {}  # chunk_id -> embedding
        self.next_id = 0
        self.deleted_ids = set()  # Track deleted IDs for reuse
        
        logger.info(f"Initialized {index_type.upper()} index with dimension {dimension}")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        if self.index_type == 'hnsw':
            if self.metric == 'cosine':
                # For cosine similarity, use IP with normalized vectors
                index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
            elif self.metric == 'l2':
                index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_L2)
            else:  # ip
                index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
            
            # Set HNSW parameters
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            
        elif self.index_type == 'ivf':
            # IVF index for larger datasets
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension) if self.metric != 'l2' else faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
        else:  # flat
            if self.metric == 'cosine' or self.metric == 'ip':
                index = faiss.IndexFlatIP(self.dimension)
            else:  # l2
                index = faiss.IndexFlatL2(self.dimension)
        
        return index
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity"""
        if self.metric == 'cosine':
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return embedding / norm
        return embedding
    
    def add_embeddings(self, embeddings: Dict[str, np.ndarray], chunks: List[Chunk]):
        """Add embeddings to FAISS index"""
        chunk_dict = {chunk.id: chunk for chunk in chunks}
        
        embeddings_list = []
        chunk_ids = []
        faiss_ids = []
        
        for chunk_id, embedding in embeddings.items():
            if chunk_id in chunk_dict and chunk_id not in self.chunk_map:
                # Normalize embedding if needed
                normalized_embedding = self._normalize_embedding(embedding)
                embeddings_list.append(normalized_embedding)
                chunk_ids.append(chunk_id)
                
                # Reuse deleted ID or create new one
                if self.deleted_ids:
                    faiss_id = self.deleted_ids.pop()
                else:
                    faiss_id = self.next_id
                    self.next_id += 1
                    
                faiss_ids.append(faiss_id)
                
                # Update mappings
                self.chunk_map[chunk_id] = chunk_dict[chunk_id]
                self.id_map[faiss_id] = chunk_id
                self.reverse_id_map[chunk_id] = faiss_id
                self.embedding_map[chunk_id] = normalized_embedding
        
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype('float32')
            
            # Train index if needed (for IVF)
            if self.index_type == 'ivf' and not self.index.is_trained:
                if len(embeddings_list) >= 100:  # Need enough data to train
                    self.index.train(embeddings_array)
                    logger.info("Trained IVF index")
                else:
                    logger.warning("Not enough data to train IVF index, need at least 100 samples")
                    return
            
            # Add to index
            if self.index_type == 'ivf' and self.index.is_trained:
                self.index.add(embeddings_array)
            elif self.index_type != 'ivf':
                self.index.add(embeddings_array)
            
            logger.info(f"Added {len(embeddings_list)} embeddings to {self.index_type.upper()} index")
    
    def search(self, query_embedding: np.ndarray, k: int, filter_dict: Dict = None) -> List[RetrievalResult]:
        """Search for similar chunks"""
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Normalize query embedding
        query_embedding = self._normalize_embedding(query_embedding)
        
        # Set search parameters for HNSW
        if self.index_type == 'hnsw':
            self.index.hnsw.efSearch = max(self.ef_search, k * 2)
        
        # Search in FAISS
        search_k = min(k * 3, self.index.ntotal)  # Get more candidates for filtering
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            search_k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            chunk_id = self.id_map.get(idx)
            if not chunk_id or chunk_id not in self.chunk_map:
                continue
            
            chunk = self.chunk_map[chunk_id]
            
            # Apply filters
            if filter_dict and not self._matches_filter(chunk, filter_dict):
                continue
            
            # Convert distance to similarity score for cosine/IP
            if self.metric in ['cosine', 'ip']:
                similarity_score = float(score)
            else:  # L2 distance - convert to similarity
                similarity_score = 1.0 / (1.0 + float(score))
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                score=similarity_score,
                chunk=chunk
            )
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID"""
        return self.chunk_map.get(chunk_id)
    
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks from the index"""
        try:
            deleted_count = 0
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_map:
                    # Get FAISS ID
                    faiss_id = self.reverse_id_map.get(chunk_id)
                    if faiss_id is not None:
                        # Remove from mappings
                        del self.chunk_map[chunk_id]
                        del self.id_map[faiss_id]
                        del self.reverse_id_map[chunk_id]
                        if chunk_id in self.embedding_map:
                            del self.embedding_map[chunk_id]
                        
                        # Mark ID as deleted for reuse
                        self.deleted_ids.add(faiss_id)
                        deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} chunks from index")
            
            # Note: FAISS doesn't support direct deletion, so we just remove from mappings
            # The actual vectors remain in the index but become inaccessible
            # For complete removal, you'd need to rebuild the index
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False
    
    def update_chunk(self, chunk_id: str, new_embedding: np.ndarray, new_chunk: Chunk):
        """Update a chunk's embedding and data"""
        if chunk_id in self.chunk_map:
            # Delete old chunk
            self.delete_chunks([chunk_id])
            
            # Add new chunk
            self.add_embeddings({chunk_id: new_embedding}, [new_chunk])
            logger.info(f"Updated chunk {chunk_id}")
        else:
            logger.warning(f"Chunk {chunk_id} not found for update")
    
    def rebuild_index(self):
        """Rebuild index to reclaim space from deleted vectors"""
        if not self.chunk_map:
            logger.warning("No chunks to rebuild index")
            return
        
        logger.info("Rebuilding index to reclaim space...")
        
        # Save current data
        chunks = list(self.chunk_map.values())
        embeddings = {chunk_id: self.embedding_map[chunk_id] 
                     for chunk_id in self.chunk_map.keys()}
        
        # Reset index and mappings
        self.index = self._create_index()
        self.chunk_map.clear()
        self.id_map.clear()
        self.reverse_id_map.clear()
        self.embedding_map.clear()
        self.next_id = 0
        self.deleted_ids.clear()
        
        # Re-add all chunks
        self.add_embeddings(embeddings, chunks)
        logger.info("Index rebuilt successfully")
    
    def save_index(self, path: str) -> bool:
        """Save index and metadata to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.index")
            
            # Save metadata
            metadata = {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'metric': self.metric,
                'm': self.m,
                'ef_construction': self.ef_construction,
                'ef_search': self.ef_search,
                'chunk_map': self.chunk_map,
                'id_map': self.id_map,
                'reverse_id_map': self.reverse_id_map,
                'embedding_map': self.embedding_map,
                'next_id': self.next_id,
                'deleted_ids': self.deleted_ids
            }
            
            with open(f"{path}.metadata", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved index to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def load_index(self, path: str) -> bool:
        """Load index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.index")
            
            # Load metadata
            with open(f"{path}.metadata", 'rb') as f:
                metadata = pickle.load(f)
            
            # Restore configuration
            self.dimension = metadata['dimension']
            self.index_type = metadata['index_type']
            self.metric = metadata['metric']
            self.m = metadata['m']
            self.ef_construction = metadata['ef_construction']
            self.ef_search = metadata['ef_search']
            
            # Restore mappings
            self.chunk_map = metadata['chunk_map']
            self.id_map = metadata['id_map']
            self.reverse_id_map = metadata['reverse_id_map']
            self.embedding_map = metadata['embedding_map']
            self.next_id = metadata['next_id']
            self.deleted_ids = metadata['deleted_ids']
            
            # Set HNSW search parameters if needed
            if self.index_type == 'hnsw':
                self.index.hnsw.efSearch = self.ef_search
            
            logger.info(f"Loaded index from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'active_chunks': len(self.chunk_map),
            'deleted_chunks': len(self.deleted_ids),
            'next_id': self.next_id
        }
    
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
        """Hierarchical retrieval: child chunks -> parent chunks"""
        # Step 1: Retrieve child chunks for precision
        child_results = self.vector_store.search(
            query_embedding,
            k=k * 3,  # Get more child candidates
            filter_dict={'chunk_type': ['child', 'table_child']}
        )
        
        # Step 2: Get parent chunks for context
        final_results = []
        seen_parents = set()
        
        for child_result in child_results[:k]:
            chunk = child_result.chunk
            
            # Get parent chunk if available
            parent_chunk = None
            if chunk.parent_id:
                parent_chunk = self.vector_store.get_chunk_by_id(chunk.parent_id)
                
                # Avoid duplicate parents
                if parent_chunk and parent_chunk.id not in seen_parents:
                    seen_parents.add(parent_chunk.id)
                    
                    # Create result with parent context
                    result = RetrievalResult(
                        chunk_id=chunk.id,
                        score=child_result.score,
                        chunk=chunk,
                        parent_chunk=parent_chunk
                    )
                    final_results.append(result)
            else:
                # Standalone chunk
                final_results.append(child_result)
            
            if len(final_results) >= k:
                break
        
        return final_results
    
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
    
    def format_context(self, results: List[RetrievalResult], max_tokens: int = 4096) -> str:
        """Format retrieval results into context for LLM"""
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(results):
            # Use parent chunk if available for better context
            chunk_to_use = result.parent_chunk if result.parent_chunk else result.chunk
            
            # Add metadata context
            metadata = chunk_to_use.metadata
            context_header = f"**Nguồn {i+1}**"
            
            if metadata.get('hierarchy_path'):
                context_header += f" ({metadata['hierarchy_path']})"
            
            context_header += f" - Độ liên quan: {result.score:.3f}"
            
            # Add content
            chunk_content = chunk_to_use.content
            chunk_tokens = estimate_tokens(chunk_content)
            
            # Check if we have space
            if current_tokens + chunk_tokens > max_tokens:
                # Truncate content to fit
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only add if meaningful space left
                    words = chunk_content.split()
                    truncated_words = words[:int(remaining_tokens * 0.8)]  # Conservative estimate
                    chunk_content = ' '.join(truncated_words) + "..."
                    chunk_tokens = remaining_tokens
                else:
                    break
            
            context_parts.append(f"{context_header}\n{chunk_content}")
            current_tokens += chunk_tokens + 20  # Header overhead
            
            if current_tokens >= max_tokens:
                break
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)