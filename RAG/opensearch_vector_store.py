# --- opensearch_vector_store.py ---

import logging
from typing import List, Dict, Optional
import numpy as np
from dataclasses import asdict

from opensearchpy import OpenSearch, NotFoundError, RequestsHttpConnection
from opensearchpy.helpers import bulk

from retriever import VectorStore, RetrievalResult
from chunker import Chunk
from config import OpenSearchConfig

logger = logging.getLogger(__name__)

class OpenSearchVectorStore(VectorStore):
    """VectorStore implementation using AWS OpenSearch Service."""

    def __init__(self, config: OpenSearchConfig, dimension: int):
        self.config = config
        self.dimension = dimension
        
        auth = self.config.http_auth
        if isinstance(auth, list): # Chuyển đổi từ list (từ JSON) sang tuple
            auth = tuple(auth)

        try:
            self.client = OpenSearch(
                hosts=[{'host': url.replace('https://', ''), 'port': 443} for url in self.config.hosts],
                http_auth=auth,
                use_ssl=self.config.use_ssl,
                verify_certs=self.config.verify_certs,
                ssl_assert_hostname=self.config.ssl_assert_hostname,
                ssl_show_warn=self.config.ssl_show_warn,
                connection_class=RequestsHttpConnection
            )
            # Kiểm tra kết nối
            if not self.client.ping():
                raise ConnectionError("Could not connect to OpenSearch")
            logger.info("Successfully connected to OpenSearch.")
            self._create_index_if_not_exists()
        except Exception as e:
            logger.error(f"Failed to connect to OpenSearch: {e}")
            raise

    def _create_index_if_not_exists(self):
        """Creates the OpenSearch index with the correct k-NN mapping if it doesn't exist."""
        if not self.client.indices.exists(index=self.config.index_name):
            logger.info(f"Index '{self.config.index_name}' not found. Creating a new one.")
            
            # Cấu hình index cho k-NN
            index_body = {
                "settings": {
                    "index.knn": True,
                    "index.knn.space_type": "cosinesimil"  # Phù hợp với embedding đã normalize
                },
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "faiss" # Hoặc 'nmslib', 'luceneknn'
                            }
                        },
                        "content": {"type": "text"},
                        "chunk_type": {"type": "keyword"},
                        "parent_id": {"type": "keyword"},
                        "document_title": {"type": "keyword"},
                        "metadata": {"type": "object", "enabled": False} # Lưu metadata nhưng không index
                    }
                }
            }
            
            try:
                self.client.indices.create(index=self.config.index_name, body=index_body)
                logger.info(f"Index '{self.config.index_name}' created successfully.")
            except Exception as e:
                logger.error(f"Failed to create index '{self.config.index_name}': {e}")
                raise

    def add_embeddings(self, embeddings: Dict[str, np.ndarray], chunks: List[Chunk]):
        """Adds embeddings and chunk data to OpenSearch using the bulk API."""
        actions = []
        for chunk in chunks:
            if chunk.id in embeddings:
                action = {
                    "_op_type": "index",
                    "_index": self.config.index_name,
                    "_id": chunk.id,
                    "_source": {
                        "embedding": embeddings[chunk.id].tolist(),
                        "content": chunk.content,
                        "chunk_type": chunk.chunk_type,
                        "parent_id": chunk.parent_id,
                        "document_title": chunk.metadata.get('document_title'),
                        "metadata": chunk.metadata,
                    }
                }
                actions.append(action)
        
        if actions:
            try:
                success, failed = bulk(self.client, actions, raise_on_error=True)
                logger.info(f"Successfully indexed {success} documents.")
                if failed:
                    logger.warning(f"Failed to index {len(failed)} documents.")
            except Exception as e:
                logger.error(f"Error during bulk indexing in OpenSearch: {e}")

    def search(self, query_embedding: np.ndarray, k: int, filter_dict: Dict = None) -> List[RetrievalResult]:
        """Performs a k-NN search on OpenSearch."""
        query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding.tolist(),
                        "k": k
                    }
                }
            }
        }
        
        # Thêm bộ lọc nếu có
        if filter_dict:
            filter_clauses = [{"term": {field: value}} for field, value in filter_dict.items()]
            knn_query = query["query"]
            query["query"] = {
                "bool": {
                    "must": [knn_query],
                    "filter": filter_clauses
                }
            }

        try:
            response = self.client.search(index=self.config.index_name, body=query)
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                # Tái tạo đối tượng Chunk từ source
                chunk_data = {
                    'id': hit['_id'],
                    'content': source.get('content', ''),
                    'chunk_type': source.get('chunk_type', ''),
                    'metadata': source.get('metadata', {}),
                    'parent_id': source.get('parent_id'),
                }
                # Thêm các trường còn thiếu với giá trị mặc định
                chunk_data['child_ids'] = source.get('metadata', {}).get('child_ids', [])
                chunk_data['tokens'] = source.get('metadata', {}).get('tokens', 0)

                reconstructed_chunk = Chunk(**chunk_data)

                results.append(RetrievalResult(
                    chunk_id=hit["_id"],
                    score=hit["_score"],
                    chunk=reconstructed_chunk
                ))
            return results
        except Exception as e:
            logger.error(f"Error during search in OpenSearch: {e}")
            return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieves a single chunk by its ID."""
        try:
            response = self.client.get(index=self.config.index_name, id=chunk_id)
            source = response["_source"]
            chunk_data = {
                'id': response['_id'],
                'content': source.get('content', ''),
                'chunk_type': source.get('chunk_type', ''),
                'metadata': source.get('metadata', {}),
                'parent_id': source.get('parent_id'),
                'child_ids': source.get('metadata', {}).get('child_ids', []),
                'tokens': source.get('metadata', {}).get('tokens', 0)
            }
            return Chunk(**chunk_data)
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id} from OpenSearch: {e}")
            return None
            
    # Các phương thức sau không áp dụng cho OpenSearch vì nó là dịch vụ managed
    def save_index(self, path: str) -> bool:
        logger.info("save_index is a no-op for OpenSearchVectorStore as it's a managed service.")
        return True

    def load_index(self, path: str) -> bool:
        logger.info("load_index is a no-op for OpenSearchVectorStore. Connection is established on init.")
        return True
        
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Deletes chunks from the index by their IDs."""
        actions = [
            {
                "_op_type": "delete",
                "_index": self.config.index_name,
                "_id": chunk_id,
            }
            for chunk_id in chunk_ids
        ]
        if actions:
            try:
                success, failed = bulk(self.client, actions, raise_on_error=True)
                logger.info(f"Successfully deleted {success} documents from OpenSearch.")
                return success > 0
            except Exception as e:
                logger.error(f"Error during bulk deletion in OpenSearch: {e}")
                return False
        return True

    def get_stats(self) -> Dict:
        """Gets statistics about the index."""
        try:
            stats = self.client.indices.stats(index=self.config.index_name, metric="docs")
            doc_count = stats["indices"][self.config.index_name]["primaries"]["docs"]["count"]
            return {
                "vector_store_type": "opensearch",
                "index_name": self.config.index_name,
                "active_chunks": doc_count
            }
        except Exception as e:
            logger.error(f"Could not get stats for index {self.config.index_name}: {e}")
            return {}