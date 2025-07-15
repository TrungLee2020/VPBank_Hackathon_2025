# --- retrieval_pipeline.py ---

import logging
import argparse
from typing import Dict, List
import json
import boto3
from botocore.exceptions import ClientError

# Import các thành phần cần thiết
from config import RAGConfig
from embedder import VietnameseEmbedder
from retriever import HierarchicalRetriever
from opensearch_vector_store import OpenSearchVectorStore
from retriever import FAISSVectorStore, VectorStore

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalPipeline:
    """
    Class to handle the retrieval, reranking, and generation of answers.
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # 1. Initialize components needed for retrieval
        self.embedder = VietnameseEmbedder(self.config.embedding)
        self.vector_store = self._create_vector_store()
        
        # 2. Initialize the retriever, which contains the reranker
        self.retriever = HierarchicalRetriever(
            self.vector_store,
            self.embedder,
            reranker_config=self.config.reranker
        )
        
        # 3. Initialize Bedrock client for generation
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime', 
                region_name=self.config.bedrock.aws_region
            )
        except Exception as e:
            logger.error(f"Could not create Bedrock client: {e}")
            self.bedrock_client = None

        logger.info(f"Retrieval pipeline initialized. Reranker is {'ENABLED' if self.config.reranker.enabled else 'DISABLED'}.")

    def _create_vector_store(self) -> VectorStore:
        if self.config.vector_store_type == 'opensearch':
            logger.info("Connecting to OpenSearch vector store...")
            return OpenSearchVectorStore(self.config.opensearch, self.config.embedding.model_dimension)
        elif self.config.vector_store_type == 'faiss':
            logger.info("Loading FAISS vector store from disk...")
            faiss_store = FAISSVectorStore(self.config.embedding.model_dimension)
            if not faiss_store.load_index('./indices/default_faiss'):
                raise FileNotFoundError("FAISS index not found at './indices/default_faiss'. Please run the indexing pipeline first.")
            return faiss_store
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.vector_store_type}")


    def _create_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Tạo prompt hoàn chỉnh cho LLM."""
        context = ""
        for i, chunk in enumerate(context_chunks):
            context += f"<source_{i+1}>\n{chunk['content']}\n</source_{i+1}>\n\n"
        
        # Sử dụng Claude-style prompt
        prompt = f"""Human: Dựa vào các tài liệu được cung cấp trong cặp thẻ <sources></sources> dưới đây để trả lời câu hỏi một cách chi tiết và chính xác.
Chỉ sử dụng thông tin từ các tài liệu đã cho. Nếu không tìm thấy thông tin, hãy nói "Tôi không tìm thấy thông tin trong tài liệu."

<sources>
{context}
</sources>

Câu hỏi: {query}

Assistant:"""
        return prompt

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Gọi Bedrock LLM để sinh câu trả lời."""
        if not self.bedrock_client:
            return "Lỗi: Bedrock client chưa được khởi tạo."
        if not context_chunks:
            return "Tôi không tìm thấy thông tin trong tài liệu."
            
        prompt = self._create_prompt(query, context_chunks)
        
        # Sử dụng Claude 3 Sonnet qua Bedrock
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "top_p": 0.9,
        })
        
        try:
            response = self.bedrock_client.invoke_model(
                body=body, 
                modelId="anthropic.claude-3-sonnet-20240229-v1:0", # Model Claude 3 Sonnet
                accept="application/json", 
                contentType="application/json"
            )
            response_body = json.loads(response.get('body').read())
            
            answer = response_body.get('content')[0].get('text')
            return answer
            
        except ClientError as e:
            logger.error(f"Lỗi khi gọi Bedrock: {e}")
            return f"Đã xảy ra lỗi khi kết nối với mô hình ngôn ngữ: {e.response['Error']['Message']}"
        except Exception as e:
            logger.error(f"Lỗi không xác định khi gọi Bedrock: {e}")
            return "Đã xảy ra lỗi không xác định."


    def search(self, query: str, k: int, strategy: str) -> Dict:
        """
        Thực hiện toàn bộ chu trình: search, retrieve, rerank, và generate.
        """
        logger.info(f"Performing full RAG search for query: '{query}'")
        
        # Bước 1 & 2: Retrieve và Rerank
        results = self.retriever.retrieve(
            query=query,
            k=k,
            retrieval_strategy=strategy
        )
        
        if not results:
            return {
                'query': query,
                'answer': 'Tôi không tìm thấy thông tin nào liên quan trong tài liệu.',
                'sources': []
            }

        # Format sources cho LLM và cho việc hiển thị
        source_chunks = []
        for res in results:
            source_chunks.append({
                'final_score': res.score,
                'chunk_id': res.chunk.id,
                'content': res.chunk.content, # Gửi nội dung chunk con chính xác cho LLM
                'metadata': res.chunk.metadata,
            })
            
        # Bước 3: Generate answer
        logger.info("Generating answer using Bedrock LLM...")
        final_answer = self.generate_answer(query, source_chunks)
        
        return {
            'query': query,
            'answer': final_answer,
            'sources': source_chunks
        }
        
def interactive_session(pipeline: RetrievalPipeline):
    """Starts an interactive command-line session for querying."""
    print("\n" + "="*60)
    print("🔍 Interactive RAG Session (with Answer Generation)")
    print("="*60)
    print("Type your question to search. Type 'quit' or 'exit' to end.")
    
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            if query.lower() in ['quit', 'exit']:
                break
            
            if query:
                # k=5 là số lượng ứng viên ban đầu cho reranker
                response = pipeline.search(query=query, k=5, strategy='hierarchical')
                
                print("\n" + "="*20 + " Final Answer " + "="*20)
                print(f"\n💬 **Answer:**\n{response['answer']}")
                
                print("\n" + "="*20 + " Sources Used " + "="*20)
                if response['sources']:
                    for i, source in enumerate(response['sources']):
                        print(f"\n--- Source {i+1} (Score: {source['final_score']:.4f}) ---")
                        print(f"  Chunk ID: {source['chunk_id']}")
                        print(f"  Content: {source['content'][:250]}...")
                else:
                    print("No sources were used.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"An error occurred during query: {e}", exc_info=True)
            print("An error occurred. Please check the logs.")
            
    print("\n👋 Goodbye!")

def main():
    parser = argparse.ArgumentParser(
        description="Retrieval Pipeline for Vietnamese RAG System. This script queries an existing vector database.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--query", help="A single query to execute. The results will be printed as JSON.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive query session.")

    parser.add_argument(
        '--vector-store-type', 
        choices=['faiss', 'opensearch'], 
        default='opensearch',
        help='The type of vector store to query.'
    )
    parser.add_argument(
        '--disable-reranker', 
        action='store_true', 
        help='Disable the reranking step.'
    )

    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.error("You must specify either --query or --interactive.")

    config = RAGConfig()
    config.vector_store_type = args.vector_store_type
    if args.disable_reranker:
        config.reranker.enabled = False

    pipeline = RetrievalPipeline(config)
    
    if args.query:
        results = pipeline.search(query=args.query, k=5, strategy='hierarchical')
        print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.interactive:
        interactive_session(pipeline)


if __name__ == "__main__":
    main()
    """
    python retrieval_pipeline.py --interactive --vector-store-type opensearch

    """