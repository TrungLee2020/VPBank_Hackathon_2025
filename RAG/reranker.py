# --- reranker.py ---

import logging
from typing import List, Tuple
from sentence_transformers import CrossEncoder
import torch
from config import RerankerConfig
from retriever import RetrievalResult

logger = logging.getLogger(__name__)

class Reranker:
    """
    Sử dụng một Cross-Encoder model để rerank các kết quả truy xuất.
    Cross-Encoder nhận cả câu query và document, sau đó đưa ra một điểm số về mức độ liên quan.
    """
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = None
        if self.config.enabled:
            self._load_model()

    def _load_model(self):
        """Tải mô hình Cross-Encoder."""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = CrossEncoder(self.config.model_name, device=device)
            logger.info(f"Loaded reranker model '{self.config.model_name}' on {device}.")
        except Exception as e:
            logger.error(f"Failed to load reranker model '{self.config.model_name}': {e}")
            # Tự động vô hiệu hóa reranking nếu không tải được model
            self.config.enabled = False
            self.model = None

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Rerank một danh sách các RetrievalResult.

        Args:
            query: Câu truy vấn gốc.
            results: Danh sách kết quả từ bước truy xuất ban đầu (từ VectorStore).

        Returns:
            Danh sách kết quả đã được sắp xếp lại và lọc theo điểm số rerank.
        """
        if not self.config.enabled or not self.model or not results:
            return results

        logger.info(f"Reranking {len(results)} results for query: '{query[:50]}...'")

        # Tạo các cặp (query, document_content) để đưa vào model
        pairs = []
        for result in results:
            # Sử dụng content của chunk con (chính xác hơn) để rerank
            pairs.append((query, result.chunk.content))

        # Tính điểm số rerank
        try:
            scores = self.model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error during reranking prediction: {e}")
            # Trả về kết quả ban đầu nếu có lỗi
            return results

        # Gán điểm số mới cho mỗi kết quả và sắp xếp lại
        for result, score in zip(results, scores):
            # Ghi đè điểm số cũ bằng điểm số từ reranker
            result.score = float(score)

        # Sắp xếp lại danh sách kết quả dựa trên điểm số rerank mới (cao đến thấp)
        results.sort(key=lambda x: x.score, reverse=True)

        # Trả về top N kết quả sau khi rerank
        return results[:self.config.top_n]