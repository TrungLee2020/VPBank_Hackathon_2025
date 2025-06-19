#!/usr/bin/env python3
"""
Demo Script cho Vietnamese RAG System
Minh họa các tính năng chính của hệ thống
"""

import asyncio
import os
import time
from pathlib import Path
from enhanced_rag_main import VietnameseRAGSystem
from vector_store_manager import VectorStoreManager
from config import RAGConfig

class RAGDemo:
    """Demo class cho Vietnamese RAG System"""
    
    def __init__(self):
        self.config = RAGConfig()
        self.demo_index_path = "./indices/demo_index"
        self.rag_system = None
        self.manager = VectorStoreManager()
    
    async def setup_demo_data(self):
        """Thiết lập dữ liệu demo"""
        print("🚀 Setting up Demo Data...")
        
        # Sample Vietnamese documents
        demo_docs = {
            "luat_doanh_nghiep.md": """
# LUẬT DOANH NGHIỆP

## Chương 1: NHỮNG QUY ĐỊNH CHUNG

### Điều 1. Phạm vi điều chỉnh
Luật này quy định về thành lập, tổ chức quản lý, tổ chức lại và giải thể doanh nghiệp; quyền và nghĩa vụ của doanh nghiệp, thành viên, chủ sở hữu, người quản lý doanh nghiệp.

### Điều 2. Đối tượng áp dụng
1. Doanh nghiệp được thành lập và hoạt động theo Luật này
2. Tổ chức, cá nhân có liên quan đến việc thành lập và hoạt động của doanh nghiệp

### Điều 3. Các loại hình doanh nghiệp
1. Công ty trách nhiệm hữu hạn
2. Công ty cổ phần
3. Công ty hợp danh
4. Doanh nghiệp tư nhân

## Chương 2: THÀNH LẬP DOANH NGHIỆP

### Điều 10. Điều kiện thành lập doanh nghiệp
Tổ chức, cá nhân có quyền thành lập doanh nghiệp theo quy định của Luật này và pháp luật có liên quan.

| Loại hình | Vốn tối thiểu | Số thành viên |
|-----------|---------------|---------------|
| TNHH một thành viên | Không quy định | 1 |
| TNHH hai thành viên trở lên | Không quy định | 2-50 |
| Cổ phần | Không quy định | Tối thiểu 3 |
| Hợp danh | Không quy định | Tối thiểu 2 |
            """,
            
            "thong_tu_bo_cong_thuong.md": """
# THÔNG TƯ HƯỚNG DẪN THỰC HIỆN

## Điều 1. Phạm vi điều chỉnh
Thông tư này hướng dẫn thực hiện một số điều của Luật Doanh nghiệp về:
- Thủ tục thành lập doanh nghiệp
- Nội dung hồ sơ đăng ký kinh doanh
- Thời gian xử lý hồ sơ

## Điều 2. Hồ sơ đăng ký doanh nghiệp
### 2.1. Hồ sơ gồm:
1. Giấy đề nghị đăng ký doanh nghiệp
2. Điều lệ công ty (đối với công ty TNHH, công ty cổ phần)
3. Bản sao căn cứ pháp lý của trụ sở
4. Bản sao giấy chứng minh nhân dân của người đại diện

### 2.2. Thời gian xử lý
- Trường hợp thường: 20 ngày làm việc
- Trường hợp thuận lợi: 15 ngày làm việc

| Bước | Thời gian | Cơ quan thực hiện |
|------|-----------|-------------------|
| Tiếp nhận hồ sơ | 1 ngày | Phòng Đăng ký kinh doanh |
| Thẩm định | 15-18 ngày | Ban chuyên môn |
| Cấp giấy phép | 1-2 ngày | Lãnh đạo Sở |
            """,
            
            "huong_dan_thue.md": """
# HƯỚNG DẪN THỰC HIỆN NGHĨA VỤ THUẾ

## 1. Đăng ký thuế
Doanh nghiệp phải đăng ký thuế trong vòng 10 ngày kể từ ngày được cấp Giấy chứng nhận đăng ký doanh nghiệp.

## 2. Các loại thuế chính
### 2.1. Thuế thu nhập doanh nghiệp (TNDN)
- Mức thuế suất chuẩn: 20%
- Mức thuế suất ưu đãi: 10%, 15% (tùy điều kiện)

### 2.2. Thuế giá trị gia tăng (GTGT)
- Phương pháp khấu trừ: 10% hoặc 5%
- Phương pháp trực tiếp: 1%, 2%, 3%

### 2.3. Các khoản thuế khác
- Thuế tài nguyên
- Thuế bảo vệ môi trường
- Thuế tiêu thụ đặc biệt

## 3. Thời hạn nộp thuế
| Loại thuế | Thời hạn | Ghi chú |
|-----------|----------|---------|
| GTGT | Tháng: 20/tháng sau<br>Quý: 30/tháng đầu quý sau | Tùy doanh thu |
| TNDN | Quý: 30/tháng đầu quý sau<br>Năm: 31/3 năm sau | Tạm nộp và quyết toán |
| TNCN | 20/tháng sau | Đối với người lao động |
            """
        }
        
        # Create demo documents directory
        demo_dir = Path("./demo_docs")
        demo_dir.mkdir(exist_ok=True)
        
        for filename, content in demo_docs.items():
            with open(demo_dir / filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"✅ Created {len(demo_docs)} demo documents in ./demo_docs/")
        return demo_dir
    
    async def demonstrate_indexing(self, docs_dir: Path):
        """Demo quá trình indexing"""
        print("\n" + "="*60)
        print("📚 DEMONSTRATION: Document Indexing")
        print("="*60)
        
        # Initialize RAG system
        self.rag_system = VietnameseRAGSystem(
            config=self.config,
            index_path=self.demo_index_path
        )
        
        # Process each document
        doc_files = list(docs_dir.glob("*.md"))
        
        print(f"Processing {len(doc_files)} documents...")
        
        for doc_file in doc_files:
            print(f"\n📄 Processing: {doc_file.name}")
            
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            start_time = time.time()
            result = await self.rag_system.process_document(content, str(doc_file))
            processing_time = time.time() - start_time
            
            if result['status'] == 'success':
                print(f"   ✅ Success: {result['chunks_created']} chunks, "
                      f"{result['embeddings_generated']} embeddings "
                      f"({processing_time:.2f}s)")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Show final statistics
        stats = self.rag_system.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Documents: {stats['documents_processed']}")
        print(f"   Chunks: {stats['chunks_created']}")
        print(f"   Embeddings: {stats['embeddings_generated']}")
        print(f"   Vector Store: {stats['vector_store']['active_chunks']} active chunks")
    
    def demonstrate_querying(self):
        """Demo các truy vấn khác nhau"""
        print("\n" + "="*60)
        print("🔍 DEMONSTRATION: Query Capabilities")
        print("="*60)
        
        # Test queries
        test_queries = [
            "Điều kiện thành lập công ty TNHH là gì?",
            "Thời gian xử lý hồ sơ đăng ký doanh nghiệp",
            "Thuế suất thu nhập doanh nghiệp",
            "Hồ sơ cần thiết để đăng ký kinh doanh",
            "Các loại hình doanh nghiệp theo luật",
            "Thời hạn nộp thuế GTGT",
            "Vốn tối thiểu để thành lập công ty cổ phần"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*20} Query {i} {'='*20}")
            print(f"❓ Question: {query}")
            
            start_time = time.time()
            result = self.rag_system.query(query, k=3, strategy='hierarchical')
            query_time = time.time() - start_time
            
            print(f"⏱️  Query time: {query_time:.3f}s")
            print(f"🎯 Confidence: {result['confidence']:.3f}")
            print(f"📋 Results found: {len(result['sources'])}")
            
            # Show top results
            for j, source in enumerate(result['sources'][:2], 1):
                print(f"\n   {j}. Score: {source['score']:.4f} | Type: {source['chunk_type']}")
                print(f"      Path: {source['hierarchy_path']}")
                print(f"      Preview: {source['content_preview']}")
            
            if len(result['sources']) > 2:
                print(f"      ... and {len(result['sources']) - 2} more results")
    
    def demonstrate_index_management(self):
        """Demo quản lý index"""
        print("\n" + "="*60)
        print("🗂️  DEMONSTRATION: Index Management")
        print("="*60)
        
        # Show current indices
        indices = self.manager.list_indices()
        print(f"\n📊 Current indices: {len(indices)}")
        
        for idx in indices:
            print(f"   • {idx['name']}: {idx['size_mb']} MB, "
                  f"{idx.get('chunks_created', '?')} chunks")
        
        # Save current index
        print(f"\n💾 Saving index...")
        self.rag_system.save_index()
        print(f"✅ Index saved to {self.demo_index_path}")
        
        # Backup index
        print(f"\n🔄 Creating backup...")
        backup_success = self.manager.backup_index("demo_index")
        
        # Show index stats
        print(f"\n📈 Index statistics:")
        stats = self.rag_system.get_statistics()
        vector_stats = stats['vector_store']
        
        for key, value in vector_stats.items():
            print(f"   {key}: {value}")
    
    def demonstrate_advanced_features(self):
        """Demo các tính năng nâng cao"""
        print("\n" + "="*60)
        print("🚀 DEMONSTRATION: Advanced Features")
        print("="*60)
        
        # Test different retrieval strategies
        query = "Thủ tục thành lập doanh nghiệp"
        strategies = ['hierarchical', 'table_aware', 'simple']
        
        print(f"🔍 Testing retrieval strategies with query: '{query}'")
        
        for strategy in strategies:
            print(f"\n--- Strategy: {strategy} ---")
            
            start_time = time.time()
            result = self.rag_system.query(query, k=3, strategy=strategy)
            query_time = time.time() - start_time
            
            print(f"Time: {query_time:.3f}s | Confidence: {result['confidence']:.3f} | "
                  f"Results: {len(result['sources'])}")
            
            if result['sources']:
                top_result = result['sources'][0]
                print(f"Top result type: {top_result['chunk_type']} "
                      f"(score: {top_result['score']:.4f})")
        
        # Test filtering
        print(f"\n🔽 Testing content filtering...")
        
        # Query with different chunk types
        all_results = self.rag_system.retriever.vector_store.search(
            self.rag_system.embedder.embed_query(query), 
            k=10
        )
        
        chunk_types = {}
        for result in all_results:
            chunk_type = result.chunk.chunk_type
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        print("Chunk type distribution:")
        for chunk_type, count in chunk_types.items():
            print(f"   {chunk_type}: {count}")
    
    async def run_full_demo(self):
        """Chạy demo đầy đủ"""
        print("🎯 VIETNAMESE RAG SYSTEM - FULL DEMONSTRATION")
        print("="*60)
        
        try:
            # Setup
            docs_dir = await self.setup_demo_data()
            
            # Indexing
            await self.demonstrate_indexing(docs_dir)
            
            # Querying
            self.demonstrate_querying()
            
            # Index management
            self.demonstrate_index_management()
            
            # Advanced features
            self.demonstrate_advanced_features()
            
            print("\n" + "="*60)
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\n💡 Next steps:")
            print("   • Run interactive mode: python enhanced_rag_main.py --interactive")
            print("   • Manage indices: python vector_store_manager.py list")
            print("   • Add your documents: python enhanced_rag_main.py --documents-dir YOUR_DIR")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()

def quick_test():
    """Test nhanh với index có sẵn"""
    print("🔬 QUICK TEST MODE")
    print("="*30)
    
    try:
        # Try to load existing demo index
        rag_system = VietnameseRAGSystem(
            config=RAGConfig(),
            index_path="./indices/demo_index"
        )
        
        stats = rag_system.get_statistics()
        if stats['vector_store']['active_chunks'] == 0:
            print("❌ No existing index found. Run full demo first:")
            print("   python demo_script.py --full")
            return
        
        print(f"✅ Loaded index with {stats['vector_store']['active_chunks']} chunks")
        
        # Quick queries
        test_queries = [
            "Thời gian xử lý hồ sơ đăng ký doanh nghiệp",
            "Thuế suất doanh nghiệp",
            "Điều kiện thành lập công ty"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            result = rag_system.query(query, k=2)
            print(f"   Confidence: {result['confidence']:.3f}")
            if result['sources']:
                print(f"   Top result: {result['sources'][0]['content_preview']}")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Vietnamese RAG System Demo')
    parser.add_argument('--full', action='store_true', help='Run full demonstration')
    parser.add_argument('--quick', action='store_true', help='Quick test with existing index')
    
    args = parser.parse_args()
    
    if args.full:
        demo = RAGDemo()
        asyncio.run(demo.run_full_demo())
    elif args.quick:
        quick_test()
    else:
        print("Vietnamese RAG System Demo")
        print("=" * 30)
        print("Options:")
        print("  --full   Run full demonstration (setup + indexing + querying)")
        print("  --quick  Quick test with existing index")
        print("\nExample:")
        print("  python demo_script.py --full")

if __name__ == "__main__":
    main()