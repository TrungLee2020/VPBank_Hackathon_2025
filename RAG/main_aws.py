# --- main_aws.py ---

from typing import List, Dict, Tuple, Optional
from config import RAGConfig
from preprocessor import VietnameseMarkdownPreprocessor
from chunker import HierarchicalChunker
from table_processor import TableProcessor
from embedder import VietnameseEmbedder
from retriever import HierarchicalRetriever, RetrievalResult, VectorStore, FAISSVectorStore
# Import các lớp tokenizer mới
from utils import Tokenizer, HeuristicTokenizer, BedrockTokenizer
from opensearch_vector_store import OpenSearchVectorStore

import os
import json
import logging  
import asyncio
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VietnameseRAGSystem:
    """Enhanced RAG system với vector store management"""
    
    def __init__(self, config: RAGConfig = None, index_path: str = None, use_bedrock_tokenizer: bool = False):
        self.config = config or RAGConfig()
        self.index_path = index_path or "./indices/default_index"
        
        # Initialize tokenizer
        if use_bedrock_tokenizer:
            logger.info("Using Bedrock Tokenizer for accurate token counting.")
            self.tokenizer = BedrockTokenizer(self.config.bedrock)
        else:
            logger.info("Using Heuristic Tokenizer for token estimation.")
            self.tokenizer = HeuristicTokenizer()
            
        # Initialize components
        self.preprocessor = VietnameseMarkdownPreprocessor()
        # Truyền tokenizer vào chunker
        self.chunker = HierarchicalChunker(self.config.chunking, self.tokenizer)
        self.table_processor = TableProcessor(self.config.table)
        self.embedder = VietnameseEmbedder(self.config.embedding)
        
        # Initialize vector store
        self.vector_store = self._create_vector_store()
        self.retriever = HierarchicalRetriever(self.vector_store, self.embedder)
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'last_updated': None
        }
        
        # Try to load existing index
        self._try_load_existing_index()

    def _create_vector_store(self) -> VectorStore:
        """Create vector store based on configuration"""
        if self.config.vector_store_type == 'faiss':
            return FAISSVectorStore(
                dimension=self.config.embedding.model_dimension,
                index_type='hnsw',
                m=32,
                ef_construction=200,
                ef_search=128,
                metric='cosine'
            )
        # Opensearch config
        elif self.config.vector_store_type == 'opensearch':
            logger.info("Creating OpenSearch vector store.")
            return OpenSearchVectorStore(
                config=self.config.opensearch,
                dimension=self.config.embedding.model_dimension
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.vector_store_type}")
    
    def _try_load_existing_index(self):
        """Try to load existing index if available"""
        if os.path.exists(f"{self.index_path}.index"):
            try:
                success = self.vector_store.load_index(self.index_path)
                if success:
                    # Load stats
                    stats_path = f"{self.index_path}.stats"
                    if os.path.exists(stats_path):
                        with open(stats_path, 'r', encoding='utf-8') as f:
                            self.stats.update(json.load(f))
                    
                    logger.info(f"Loaded existing index from {self.index_path}")
                    logger.info(f"Vector store stats: {self.vector_store.get_stats()}")
                else:
                    logger.warning("Failed to load existing index")
            except Exception as e:
                logger.error(f"Error loading existing index: {e}")
    
    async def process_document(self, content: str, document_id: str = None) -> Dict:
        """Process a single document through the full pipeline"""
        try:
            logger.info(f"Processing document {document_id}")
            
            # Step 1: Preprocess document
            logger.info("  - Preprocessing document")
            doc_structure = self.preprocessor.preprocess_document(content)
            
            # Step 2: Process tables
            logger.info("  - Processing tables")
            for i, table in enumerate(doc_structure.tables):
                doc_structure.tables[i] = self.table_processor.process_table(table)
            
            # Step 3: Create chunks
            logger.info("  - Creating hierarchical chunks")
            chunks = self.chunker.chunk_document(doc_structure)
            
            # Step 4: Generate embeddings
            logger.info("  - Generating embeddings")
            embeddings = await self.embedder.embed_chunks_async(chunks)
            
            # Step 5: Store in vector database
            logger.info("  - Storing in vector database")
            self.vector_store.add_embeddings(embeddings, chunks)
            
            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            self.stats['embeddings_generated'] += len(embeddings)
            self.stats['last_updated'] = str(asyncio.get_event_loop().time())
            
            result = {
                'status': 'success',
                'document_id': document_id,
                'chunks_created': len(chunks),
                'embeddings_generated': len(embeddings),
                'tables_processed': len(doc_structure.tables)
            }
            
            logger.info(f"Successfully processed document {document_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            return {
                'status': 'error',
                'document_id': document_id,
                'error': str(e)
            }
    
    async def process_documents_batch(self, documents: List[Tuple[str, str]]) -> List[Dict]:
        """Process multiple documents in batch"""
        logger.info(f"Processing batch of {len(documents)} documents")
        results = []
        
        for content, doc_id in documents:
            result = await self.process_document(content, doc_id)
            results.append(result)
            
            # Auto-save after each document
            if result['status'] == 'success':
                self.save_index()
        
        return results
    
    def query(self, question: str, k: int = 5, strategy: str = 'hierarchical') -> Dict:
        """Query the RAG system"""
        try:
            logger.info(f"Querying: '{question}' with strategy '{strategy}'")
            
            # Retrieve relevant chunks
            results = self.retriever.retrieve(
                query=question,
                k=k,
                retrieval_strategy=strategy
            )
            
            if not results:
                return {
                    'answer': 'Không tìm thấy thông tin liên quan đến câu hỏi.',
                    'sources': [],
                    'confidence': 0.0,
                    'context': ''
                }
            
            # Format context
            context = self.retriever.format_context(results)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(results)
            
            # Prepare sources
            sources = []
            for result in results:
                source_info = {
                    'chunk_id': result.chunk_id,
                    'score': result.score,
                    'title': result.chunk.metadata.get('section_title', 'Unknown'),
                    'hierarchy_path': result.chunk.metadata.get('hierarchy_path', ''),
                    'chunk_type': result.chunk.chunk_type,
                    'content_preview': result.chunk.content[:4098] #+ "..." if len(result.chunk.content) > 100 else result.chunk.content
                }
                sources.append(source_info)
            
            query_result = {
                'context': context,
                'sources': sources,
                'confidence': confidence,
                'strategy_used': strategy,
                'total_results': len(results)
            }
            
            logger.info(f"Query completed. Found {len(results)} results with confidence {confidence:.3f}")
            return query_result
            
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return {
                'answer': f'Lỗi khi xử lý câu hỏi: {str(e)}',
                'sources': [],
                'confidence': 0.0,
                'context': '',
                'error': str(e)
            }
    
    def _calculate_confidence(self, results: List[RetrievalResult]) -> float:
        """Calculate confidence score based on retrieval results"""
        if not results:
            return 0.0
        
        # Average of top scores with weighted decay
        weights = [0.4, 0.3, 0.2, 0.1]  # Weight for top 4 results
        weighted_score = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(results[:4]):
            weight = weights[i] if i < len(weights) else 0.05
            weighted_score += result.score * weight
            total_weight += weight
        
        return min(weighted_score / total_weight if total_weight > 0 else 0.0, 1.0)
    
    def save_index(self):
        """Save vector index and stats to disk"""
        try:
            # Create directory if not exists
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save vector store
            success = self.vector_store.save_index(self.index_path)
            
            if success:
                # Save stats
                stats_path = f"{self.index_path}.stats"
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(self.stats, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Index and stats saved to {self.index_path}")
            else:
                logger.error("Failed to save index")
                
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def load_index(self, index_path: str = None):
        """Load vector index from disk"""
        path = index_path or self.index_path
        try:
            success = self.vector_store.load_index(path)
            
            if success:
                # Load stats
                stats_path = f"{path}.stats"
                if os.path.exists(stats_path):
                    with open(stats_path, 'r', encoding='utf-8') as f:
                        self.stats.update(json.load(f))
                
                logger.info(f"Index loaded from {path}")
                return True
            else:
                logger.error(f"Failed to load index from {path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        vector_stats = self.vector_store.get_stats()
        return {
            **self.stats,
            'vector_store': vector_stats,
            'index_path': self.index_path
        }
    
    def rebuild_index(self):
        """Rebuild vector index to optimize space"""
        logger.info("Rebuilding vector index...")
        self.vector_store.rebuild_index()
        self.save_index()
        logger.info("Index rebuilt and saved")
    
    def delete_documents(self, chunk_ids: List[str]) -> bool:
        """Delete documents from the system"""
        success = self.vector_store.delete_chunks(chunk_ids)
        if success:
            self.save_index()
            logger.info(f"Deleted {len(chunk_ids)} chunks")
        return success

def interactive_query_session(rag_system: VietnameseRAGSystem):
    """Interactive query session for testing"""
    print("\n" + "="*60)
    print("🔍 INTERACTIVE QUERY SESSION")
    print("="*60)
    print("Commands:")
    print("  - Type your question to search")
    print("  - 'stats' to show statistics")
    print("  - 'rebuild' to rebuild index")
    print("  - 'save' to save index")
    print("  - 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'stats':
                stats = rag_system.get_statistics()
                print("\n📊 System Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            elif query.lower() == 'rebuild':
                rag_system.rebuild_index()
                print("✅ Index rebuilt successfully")
            elif query.lower() == 'save':
                rag_system.save_index()
                print("✅ Index saved successfully")
            elif query:
                result = rag_system.query(query, k=5, strategy='hierarchical')
                
                print(f"\n📋 Results (Confidence: {result['confidence']:.3f}):")
                print("-" * 50)
                
                for i, source in enumerate(result['sources'][:5]):
                    print(f"\n{i+1}. Score: {source['score']:.4f}")
                    print(f"   Type: {source['chunk_type']}")
                    print(f"   Path: {source['hierarchy_path']}")
                    print(f"   Preview: {source['content_preview']}")
                    print(f"   Title: {source['title']}")
                    print(f"   Chunk ID: {source['chunk_id']}")
                
                # if len(result['sources']) > 3:
                #     print(f"\n... and {len(result['sources']) - 3} more results")
                # print(f"\n... and {len(result['sources']) - 3} more results")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")

async def main():
    """Enhanced main function with comprehensive options"""
    parser = argparse.ArgumentParser(description='Vietnamese RAG System')
    parser.add_argument('--index-path', default='./indices/vietnamese_rag', 
                       help='Path to save/load vector index')
    parser.add_argument('--document-path', help='Path to document to process')
    parser.add_argument('--documents-dir', help='Directory containing documents to process')
    parser.add_argument('--query', help='Query to execute')
    parser.add_argument('--interactive', action='store_true', 
                       help='Start interactive query session')
    parser.add_argument('--rebuild', action='store_true', 
                       help='Rebuild vector index')
    parser.add_argument('--stats', action='store_true', 
                       help='Show system statistics')
    # Thêm tùy chọn sử dụng Bedrock tokenizer
    parser.add_argument('--use-bedrock-tokenizer', action='store_true', 
                       help='Use Bedrock for accurate token counting. Requires AWS credentials.')
    parser.add_argument(
        '--vector-store-type', 
        choices=['faiss', 'opensearch'], 
        default='opensearch',
        help='The type of vector store to use.'
    )
    args = parser.parse_args()
    
    # Initialize RAG system
    print("🚀 Initializing Vietnamese RAG System...")
    config = RAGConfig()
    rag_system = VietnameseRAGSystem(
        config, 
        args.index_path, 
        use_bedrock_tokenizer=args.use_bedrock_tokenizer
    )
    
    # Show initial stats
    initial_stats = rag_system.get_statistics()
    print(f"📊 Initial stats: {initial_stats['vector_store']['active_chunks']} chunks loaded")
    
    # Process documents if specified
    if args.document_path:
        print(f"📄 Processing document: {args.document_path}")
        with open(args.document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = await rag_system.process_document(content, args.document_path)
        print(f"✅ Processing result: {result}")
        
    elif args.documents_dir:
        print(f"📁 Processing documents from: {args.documents_dir}")
        documents = []
        
        for file_path in Path(args.documents_dir).glob("*.md"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents.append((content, str(file_path)))
        
        if documents:
            results = await rag_system.process_documents_batch(documents)
            successful = sum(1 for r in results if r['status'] == 'success')
            print(f"✅ Processed {successful}/{len(documents)} documents successfully")
        else:
            print("❌ No .md files found in directory")
    
    # Execute query if specified
    if args.query:
        print(f"🔍 Executing query: {args.query}")
        result = rag_system.query(args.query)
        
        print(f"\n📋 Query Results (Confidence: {result['confidence']:.3f}):")
        for i, source in enumerate(result['sources'][:3]):
            print(f"\n{i+1}. Score: {source['score']:.4f}")
            print(f"   Content: {source['content_preview']}")
    
    # Rebuild index if requested
    if args.rebuild:
        print("🔄 Rebuilding index...")
        rag_system.rebuild_index()
        print("✅ Index rebuilt")
    
    # Show stats if requested
    if args.stats:
        stats = rag_system.get_statistics()
        print("\n📊 System Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Start interactive session if requested
    if args.interactive:
        interactive_query_session(rag_system)
    
    # Default behavior - process sample document and start interactive session
    if not any([args.document_path, args.documents_dir, args.query, 
               args.rebuild, args.stats, args.interactive]):
        
        # Check if we have any data
        stats = rag_system.get_statistics()
        if stats['vector_store']['active_chunks'] == 0:
            # Try to process default document
            default_doc = 'output.md'
            if os.path.exists(default_doc):
                print(f"📄 Processing default document: {default_doc}")
                with open(default_doc, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                result = await rag_system.process_document(content, default_doc)
                print(f"✅ Processing result: {result}")
            else:
                print(f"❌ Default document {default_doc} not found")
                print("💡 Use --document-path or --documents-dir to add documents")
        
        # Start interactive session
        interactive_query_session(rag_system)

if __name__ == "__main__":
    asyncio.run(main())