# --- START OF FILE reranker.py ---

import logging
from typing import List
import boto3
from botocore.exceptions import ClientError

from config import RerankerConfig
from retriever import RetrievalResult

logger = logging.getLogger(__name__)

class BedrockReranker:
    """
    Sử dụng API 'rerank' của Amazon Bedrock với mô hình Cohere Rerank.
    """
    def __init__(self, config: RerankerConfig, bedrock_region: str):
        self.config = config
        self.bedrock_client = None
        if self.config.enabled:
            try:
                self.bedrock_client = boto3.client('bedrock-runtime', region_name=bedrock_region)
                logger.info(f"Initialized Bedrock Reranker client for model '{self.config.bedrock_model_id}' in region '{bedrock_region}'.")
            except Exception as e:
                logger.error(f"Failed to create Bedrock client for reranker: {e}")
                self.config.enabled = False

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Rerank một danh sách các kết quả bằng Cohere Rerank trên Bedrock.

        Args:
            query: Câu truy vấn gốc.
            results: Danh sách kết quả từ VectorStore.

        Returns:
            Danh sách kết quả đã được sắp xếp lại và lọc theo điểm của Bedrock.
        """
        if not self.config.enabled or not self.bedrock_client or not results:
            return results[:self.config.top_n]

        logger.info(f"Reranking {len(results)} results using Bedrock model '{self.config.bedrock_model_id}'...")

        # Chuẩn bị danh sách các documents (dạng chuỗi) cho API
        documents = [res.chunk.content for res in results]

        try:
            # Gọi API rerank của Bedrock
            response = self.bedrock_client.rerank(
                modelId=self.config.bedrock_model_id,
                query=query,
                documents=documents,
                topN=self.config.top_n
            )
            reranked_api_results = response.get('results', [])
        except ClientError as e:
            logger.error(f"Error calling Bedrock Rerank API: {e}")
            # Nếu lỗi, trả về top N kết quả ban đầu
            return sorted(results, key=lambda x: x.score, reverse=True)[:self.config.top_n]

        # Xử lý kết quả trả về từ API
        final_results = []
        for reranked_item in reranked_api_results:
            # Lấy chỉ số của document trong danh sách gốc
            original_index = reranked_item['index']
            new_score = reranked_item['relevance_score']

            # Lấy đối tượng RetrievalResult gốc tương ứng
            original_result = results[original_index]
            
            # Cập nhật điểm số của nó với điểm từ Cohere
            original_result.score = new_score
            
            final_results.append(original_result)
        
        # API đã trả về danh sách được sắp xếp và lọc theo topN,
        # nên không cần sắp xếp lại ở đây.
        return final_results