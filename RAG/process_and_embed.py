# --- process_and_embed.py ---

import asyncio
import argparse
import os
import json
import logging
from pathlib import Path
from dataclasses import asdict
import numpy as np

# Import các thành phần cần thiết từ hệ thống RAG
from config import RAGConfig
from preprocessor import VietnameseMarkdownPreprocessor
from table_processor import TableProcessor
from chunker import HierarchicalChunker
from embedder import VietnameseEmbedder
from utils import BedrockTokenizer

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def process_single_document(input_path: str, output_dir: str):
    """
    Xử lý một tài liệu duy nhất: Tiền xử lý -> Chunk -> Embed -> Lưu kết quả.
    Sử dụng Bedrock Tokenizer để chia chunk.
    """
    logger.info(f"🚀 Starting processing for document: {input_path}")
    
    # --- 1. Khởi tạo các thành phần ---
    config = RAGConfig()
    
    # Luôn sử dụng Bedrock Tokenizer cho script này
    logger.info("Initializing Bedrock Tokenizer for accurate chunking...")
    tokenizer = BedrockTokenizer(config.bedrock)
    
    preprocessor = VietnameseMarkdownPreprocessor()
    table_processor = TableProcessor(config.table)
    # Truyền Bedrock Tokenizer vào chunker
    chunker = HierarchicalChunker(config.chunking, tokenizer)
    embedder = VietnameseEmbedder(config.embedding)
    
    # --- 2. Đọc và tiền xử lý tài liệu ---
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info("Step 1: Preprocessing document structure...")
    doc_structure = preprocessor.preprocess_document(content)
    
    logger.info("Step 2: Processing tables...")
    for i, table in enumerate(doc_structure.tables):
        doc_structure.tables[i] = table_processor.process_table(table)
    
    # --- 3. Chia chunks ---
    logger.info("Step 3: Creating hierarchical chunks using Bedrock Tokenizer...")
    chunks = chunker.chunk_document(doc_structure)
    logger.info(f"   => Created {len(chunks)} chunks.")
    
    # --- 4. Tạo embeddings ---
    logger.info("Step 4: Generating embeddings for all chunks...")
    embeddings = await embedder.embed_chunks_async(chunks)
    logger.info(f"   => Generated {len(embeddings)} embeddings.")
    
    # --- 5. Lưu kết quả để kiểm tra ---
    os.makedirs(output_dir, exist_ok=True)
    doc_name = Path(input_path).stem
    
    # Lưu các chunks vào file JSON
    chunks_output_path = os.path.join(output_dir, f"{doc_name}_chunks.json")
    logger.info(f"Step 5a: Saving chunks to {chunks_output_path}")
    
    # Chuyển đổi list các đối tượng Chunk thành list các dictionary
    chunks_as_dicts = [asdict(chunk) for chunk in chunks]
    with open(chunks_output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_as_dicts, f, ensure_ascii=False, indent=2)
    
    # Lưu các embeddings vào file .npz (numpy)
    embeddings_output_path = os.path.join(output_dir, f"{doc_name}_embeddings.npz")
    logger.info(f"Step 5b: Saving embeddings to {embeddings_output_path}")
    np.savez_compressed(embeddings_output_path, **embeddings)
    
    logger.info(f"✅ Processing complete for {input_path}.")
    logger.info(f"   Chunks and embeddings saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Process a Vietnamese document, create chunks and embeddings, and save them for verification.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to the input Markdown document."
    )
    parser.add_argument(
        "--output-dir", 
        default="./verification_output", 
        help="Directory to save the output chunks (JSON) and embeddings (NPZ)."
    )
    
    args = parser.parse_args()
    
    # Kiểm tra điều kiện cần thiết
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist at '{args.input}'")
        return
        
    print("--- Vietnamese RAG - Processing and Embedding Verification ---")
    print(f"Input Document: {args.input}")
    print(f"Output Directory: {args.output_dir}")
    print("NOTE: This script requires AWS credentials configured to use Bedrock Tokenizer.")
    print("-" * 60)
    
    asyncio.run(process_single_document(args.input, args.output_dir))

if __name__ == "__main__":
    main()