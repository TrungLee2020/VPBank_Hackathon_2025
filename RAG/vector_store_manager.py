#!/usr/bin/env python3
"""
Vector Store Management Utilities
Công cụ quản lý vector store cho hệ thống RAG tiếng Việt
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import asyncio
from main import VietnameseRAGSystem
from config import RAGConfig

class VectorStoreManager:
    """Quản lý vector store"""
    
    def __init__(self, base_path: str = "./indices"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def list_indices(self) -> List[Dict]:
        """Liệt kê tất cả indices có sẵn"""
        indices = []
        
        for index_file in self.base_path.glob("*.index"):
            index_name = index_file.stem
            metadata_file = self.base_path / f"{index_name}.metadata"
            stats_file = self.base_path / f"{index_name}.stats"
            
            info = {
                'name': index_name,
                'path': str(index_file.parent / index_name),
                'size_mb': round(index_file.stat().st_size / (1024*1024), 2),
                'has_metadata': metadata_file.exists(),
                'has_stats': stats_file.exists()
            }
            
            # Load stats if available
            if stats_file.exists():
                try:
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                    info.update({
                        'documents_processed': stats.get('documents_processed', 0),
                        'chunks_created': stats.get('chunks_created', 0),
                        'last_updated': stats.get('last_updated', 'Unknown')
                    })
                except Exception as e:
                    info['stats_error'] = str(e)
            
            indices.append(info)
        
        return sorted(indices, key=lambda x: x['name'])
    
    def inspect_index(self, index_name: str) -> Dict:
        """Kiểm tra chi tiết một index"""
        index_path = self.base_path / index_name
        
        if not (self.base_path / f"{index_name}.index").exists():
            return {'error': f'Index {index_name} not found'}
        
        try:
            # Load RAG system with this index
            rag_system = VietnameseRAGSystem(
                config=RAGConfig(),
                index_path=str(index_path)
            )
            
            return rag_system.get_statistics()
            
        except Exception as e:
            return {'error': f'Failed to inspect index: {str(e)}'}
    
    def backup_index(self, index_name: str, backup_dir: str = None) -> bool:
        """Sao lưu index"""
        import shutil
        from datetime import datetime
        
        source_path = self.base_path / index_name
        backup_dir = backup_dir or "./backups"
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{index_name}_{timestamp}"
        
        try:
            # Copy all related files
            for suffix in ['.index', '.metadata', '.stats']:
                source_file = self.base_path / f"{index_name}{suffix}"
                if source_file.exists():
                    backup_file = backup_path / f"{backup_name}{suffix}"
                    shutil.copy2(source_file, backup_file)
            
            print(f"✅ Backup created: {backup_name}")
            return True
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def merge_indices(self, source_indices: List[str], target_name: str) -> bool:
        """Gộp nhiều indices thành một"""
        try:
            # Create new RAG system for target
            target_path = self.base_path / target_name
            target_rag = VietnameseRAGSystem(
                config=RAGConfig(),
                index_path=str(target_path)
            )
            
            # Load and merge each source index
            for source_name in source_indices:
                source_path = self.base_path / source_name
                
                if not (self.base_path / f"{source_name}.index").exists():
                    print(f"⚠️  Index {source_name} not found, skipping")
                    continue
                
                # Load source RAG system
                source_rag = VietnameseRAGSystem(
                    config=RAGConfig(),
                    index_path=str(source_path)
                )
                
                # Get all chunks from source
                source_chunks = list(source_rag.vector_store.chunk_map.values())
                source_embeddings = source_rag.vector_store.embedding_map
                
                # Add to target
                target_rag.vector_store.add_embeddings(source_embeddings, source_chunks)
                
                print(f"✅ Merged {len(source_chunks)} chunks from {source_name}")
            
            # Save merged index
            target_rag.save_index()
            print(f"✅ Merged index saved as {target_name}")
            return True
            
        except Exception as e:
            print(f"❌ Merge failed: {e}")
            return False

def print_indices_table(indices: List[Dict]):
    """In bảng thông tin indices"""
    if not indices:
        print("📭 No indices found")
        return
    
    print("\n📊 Available Vector Indices:")
    print("=" * 80)
    print(f"{'Name':<20} {'Size (MB)':<10} {'Docs':<6} {'Chunks':<8} {'Last Updated':<15}")
    print("-" * 80)
    
    for idx in indices:
        name = idx['name'][:19]
        size = idx['size_mb']
        docs = idx.get('documents_processed', '?')
        chunks = idx.get('chunks_created', '?')
        updated = idx.get('last_updated', 'Unknown')[:14]
        
        print(f"{name:<20} {size:<10} {docs:<6} {chunks:<8} {updated:<15}")
    
    print("=" * 80)

async def process_documents_to_index(docs_dir: str, index_name: str) -> bool:
    """Xử lý tất cả documents trong thư mục vào index"""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"❌ Directory {docs_dir} not found")
        return False
    
    # Find all markdown files
    md_files = list(docs_path.glob("**/*.md"))
    if not md_files:
        print(f"❌ No .md files found in {docs_dir}")
        return False
    
    print(f"📁 Found {len(md_files)} markdown files")
    
    # Initialize RAG system
    rag_system = VietnameseRAGSystem(
        config=RAGConfig(),
        index_path=f"./indices/{index_name}"
    )
    
    # Process each file
    successful = 0
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = await rag_system.process_document(content, str(md_file))
            if result['status'] == 'success':
                successful += 1
                print(f"✅ {md_file.name}: {result['chunks_created']} chunks")
            else:
                print(f"❌ {md_file.name}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ {md_file.name}: {e}")
    
    print(f"\n📊 Processed {successful}/{len(md_files)} files successfully")
    return successful > 0

def main():
    parser = argparse.ArgumentParser(description='Vector Store Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all indices')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect an index')
    inspect_parser.add_argument('index_name', help='Name of index to inspect')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup an index')
    backup_parser.add_argument('index_name', help='Name of index to backup')
    backup_parser.add_argument('--backup-dir', default='./backups', help='Backup directory')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple indices')
    merge_parser.add_argument('target_name', help='Name for merged index')
    merge_parser.add_argument('sources', nargs='+', help='Source indices to merge')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents to index')
    process_parser.add_argument('docs_dir', help='Directory containing documents')
    process_parser.add_argument('index_name', help='Name for the index')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test query on index')
    test_parser.add_argument('index_name', help='Index to test')
    test_parser.add_argument('query', help='Query to test')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = VectorStoreManager()
    
    if args.command == 'list':
        indices = manager.list_indices()
        print_indices_table(indices)
    
    elif args.command == 'inspect':
        info = manager.inspect_index(args.index_name)
        print(f"\n🔍 Index: {args.index_name}")
        print("=" * 50)
        for key, value in info.items():
            print(f"{key}: {value}")
    
    elif args.command == 'backup':
        success = manager.backup_index(args.index_name, args.backup_dir)
        if not success:
            exit(1)
    
    elif args.command == 'merge':
        success = manager.merge_indices(args.sources, args.target_name)
        if not success:
            exit(1)
    
    elif args.command == 'process':
        success = asyncio.run(process_documents_to_index(args.docs_dir, args.index_name))
        if not success:
            exit(1)
    
    elif args.command == 'test':
        try:
            rag_system = VietnameseRAGSystem(
                config=RAGConfig(),
                index_path=f"./indices/{args.index_name}"
            )
            
            result = rag_system.query(args.query)
            print(f"\n🔍 Query: {args.query}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print(f"📋 Results: {len(result['sources'])}")
            
            for i, source in enumerate(result['sources'][:3]):
                print(f"\n{i+1}. Score: {source['score']:.4f}")
                print(f"   {source['content_preview']}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            exit(1)

if __name__ == "__main__":
    main()