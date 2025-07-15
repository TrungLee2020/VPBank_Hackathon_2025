# --- indexing_pipeline.py ---

import asyncio
import argparse
import os
import logging
from pathlib import Path
from typing import List, Tuple

# Import các thành phần cần thiết
from config import RAGConfig
from preprocessor import VietnameseMarkdownPreprocessor, DocumentStructure
from table_processor import TableProcessor
from chunker import HierarchicalChunker, Chunk
from embedder import VietnameseEmbedder
from utils import BedrockTokenizer, HeuristicTokenizer
from opensearch_vector_store import OpenSearchVectorStore
from retriever import FAISSVectorStore, VectorStore

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexingPipeline:
    """
    Class to orchestrate the entire document processing and indexing pipeline.
    """
    def __init__(self, config: RAGConfig, use_bedrock_tokenizer: bool = False):
        self.config = config
        
        # 1. Initialize Tokenizer
        if use_bedrock_tokenizer:
            logger.info("Using Bedrock Tokenizer for accurate token counting.")
            self.tokenizer = BedrockTokenizer(self.config.bedrock)
        else:
            logger.info("Using Heuristic Tokenizer for token estimation.")
            self.tokenizer = HeuristicTokenizer()
        
        # 2. Initialize processing components
        self.preprocessor = VietnameseMarkdownPreprocessor()
        self.table_processor = TableProcessor(self.config.table)
        self.chunker = HierarchicalChunker(self.config.chunking, self.tokenizer)
        self.embedder = VietnameseEmbedder(self.config.embedding)
        
        # 3. Initialize Vector Store
        self.vector_store = self._create_vector_store()

    def _create_vector_store(self) -> VectorStore:
        """Create vector store instance based on configuration."""
        if self.config.vector_store_type == 'opensearch':
            logger.info("Initializing OpenSearch vector store...")
            return OpenSearchVectorStore(self.config.opensearch, self.config.embedding.model_dimension)
        elif self.config.vector_store_type == 'faiss':
            logger.info("Initializing FAISS vector store...")
            return FAISSVectorStore(self.config.embedding.model_dimension)
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.vector_store_type}")

    async def process_and_index_document(self, content: str, document_id: str):
        """Processes a single document and adds it to the vector store."""
        try:
            logger.info(f"--- Processing document: {document_id} ---")
            
            # Step 1: Preprocess document to get structure
            logger.info("Step 1/4: Preprocessing document structure...")
            doc_structure = self.preprocessor.preprocess_document(content)
            
            # Step 2: Process tables within the structure
            logger.info("Step 2/4: Processing tables...")
            for i, table in enumerate(doc_structure.tables):
                doc_structure.tables[i] = self.table_processor.process_table(table)
            
            # Step 3: Create hierarchical chunks
            logger.info("Step 3/4: Creating hierarchical chunks...")
            chunks = self.chunker.chunk_document(doc_structure)
            logger.info(f"   => Created {len(chunks)} chunks.")
            
            # Step 4: Generate embeddings and add to vector store
            logger.info("Step 4/4: Generating embeddings and indexing...")
            embeddings = await self.embedder.embed_chunks_async(chunks)
            self.vector_store.add_embeddings(embeddings, chunks)
            logger.info(f"   => Successfully indexed {len(embeddings)} chunks into {self.config.vector_store_type}.")
            
            return {'status': 'success', 'document_id': document_id, 'chunks_indexed': len(chunks)}
            
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}", exc_info=True)
            return {'status': 'error', 'document_id': document_id, 'error': str(e)}

    async def process_directory(self, documents_dir: str):
        """Processes all .md files in a given directory."""
        logger.info(f"Starting to process all documents in directory: {documents_dir}")
        doc_paths = list(Path(documents_dir).glob("*.md"))
        
        if not doc_paths:
            logger.warning(f"No .md files found in '{documents_dir}'.")
            return

        results = []
        for doc_path in doc_paths:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            result = await self.process_and_index_document(content, str(doc_path))
            results.append(result)

        successful = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"--- Batch processing finished. ---")
        logger.info(f"Successfully processed and indexed {successful}/{len(doc_paths)} documents.")


def main():
    parser = argparse.ArgumentParser(
        description="Indexing Pipeline for Vietnamese RAG System. This script processes documents and stores them in a vector database.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--document-path", help="Path to a single document to process and index.")
    group.add_argument("--documents-dir", help="Path to a directory containing documents to process and index.")
    
    parser.add_argument(
        '--vector-store-type', 
        choices=['faiss', 'opensearch'], 
        default='opensearch',
        help='The type of vector store to use. Defaults to opensearch.'
    )
    parser.add_argument(
        '--use-bedrock-tokenizer', 
        action='store_true', 
        help='Use Bedrock for accurate token counting. Requires AWS credentials.'
    )
    
    args = parser.parse_args()
    
    config = RAGConfig()
    config.vector_store_type = args.vector_store_type
    
    pipeline = IndexingPipeline(config, use_bedrock_tokenizer=args.use_bedrock_tokenizer)
    
    loop = asyncio.get_event_loop()
    if args.document_path:
        with open(args.document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        loop.run_until_complete(pipeline.process_and_index_document(content, args.document_path))
    elif args.documents_dir:
        loop.run_until_complete(pipeline.process_directory(args.documents_dir))

if __name__ == "__main__":
    main()

    """
    python indexing_pipeline.py --documents-dir ./my_documents --vector-store-type opensearch --use-bedrock-tokenizer
    """