# --- app.py ---

import streamlit as st
import boto3
import os
import time
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Import các pipeline đã xây dựng
from indexing_pipeline import IndexingPipeline
from retrieval_pipeline import RetrievalPipeline
from config import RAGConfig

# Tải biến môi trường từ file .env
load_dotenv()

# --- Cấu hình và Khởi tạo ---
st.set_page_config(page_title="Vietnamese RAG System", layout="wide")

# Lấy cấu hình từ .env
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

# Khởi tạo S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION
)

# Khởi tạo RAG config
rag_config = RAGConfig()
rag_config.vector_store_type = 'opensearch'
rag_config.reranker.enabled = True 

@st.cache_resource
def get_retrieval_pipeline():
    try:
        return RetrievalPipeline(config=rag_config)
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo Retrieval Pipeline: {e}. Đảm bảo OpenSearch/Bedrock đã hoạt động và có quyền truy cập.")
        return None

retrieval_pipeline = get_retrieval_pipeline()

def upload_to_s3(file_obj, bucket, object_name):
    """Upload một file object lên S3 bucket."""
    try:
        s3_client.upload_fileobj(file_obj, bucket, object_name)
        return True
    except Exception as e:
        st.error(f"Lỗi khi tải file lên S3: {e}")
        return False

def simulate_markdown_conversion(uploaded_filename):
    """Giả lập quá trình chuyển đổi sang Markdown."""
    st.info(f"Giả lập chuyển đổi '{uploaded_filename}' sang Markdown...")
    time.sleep(2) 
    temp_dir = Path("./temp_markdowns")
    temp_dir.mkdir(exist_ok=True)
    markdown_content = f"# Tiêu đề từ file {uploaded_filename}\n\nĐây là nội dung được chuyển đổi."
    markdown_path = temp_dir / f"{Path(uploaded_filename).stem}.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    st.success("Chuyển đổi sang Markdown hoàn tất.")
    return markdown_path


async def run_indexing(markdown_path):
    """Chạy pipeline indexing và đo thời gian."""
    indexing_pipeline = IndexingPipeline(config=rag_config, use_bedrock_tokenizer=True)
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    start_time = time.time()
    with st.spinner("⏳ Đang xử lý và tạo chỉ mục cho tài liệu..."):
        result = await indexing_pipeline.process_and_index_document(content, str(markdown_path))
    end_time = time.time()
    processing_time = end_time - start_time
    
    if result['status'] == 'success':
        st.success(f"✅ Lập chỉ mục thành công!")
        st.metric(label="**Thời gian Indexing**", value=f"{processing_time:.2f} giây")
    else:
        st.error(f"❌ Lỗi khi lập chỉ mục: {result['error']}")
    os.remove(markdown_path)


# --- Giao diện Streamlit ---

st.title("Hệ thống RAG Tiếng Việt (OpenSearch + Reranker + Bedrock)")
st.markdown("---")

# Cột bên trái: Upload và Indexing
with st.sidebar:
    st.header("📚 Quản lý tài liệu")
    st.subheader("1. Tải lên tài liệu mới")
    
    uploaded_file = st.file_uploader(
        "Chọn file (PDF, DOCX, TXT, EXCEL, CSV, PNG...)", 
        type=['pdf', 'docx', 'txt', 'xlsx', 'csv', 'png', 'PNG']
    )

    if uploaded_file is not None:
        st.write(f"Đã chọn: `{uploaded_file.name}`")
        if st.button("Bắt đầu xử lý"):
            with st.status("**Đang xử lý file...**", expanded=True) as status:
                st.write("Bước 1: Tải file lên S3...")
                upload_success = upload_to_s3(uploaded_file, S3_BUCKET_NAME, f"uploads/{uploaded_file.name}")
                if not upload_success:
                    status.update(label="Tải lên thất bại!", state="error", expanded=True)
                else:
                    st.write("   => Tải lên S3 thành công.")
                    st.write("Bước 2: Chuyển đổi sang Markdown...")
                    markdown_file_path = simulate_markdown_conversion(uploaded_file.name)
                    st.write(f"   => File Markdown được tạo tại: `{markdown_file_path}`")
                    st.write("Bước 3: Lập chỉ mục tài liệu...")
                    asyncio.run(run_indexing(markdown_file_path))
                    status.update(label="Hoàn tất xử lý!", state="complete", expanded=True)
                    st.balloons()

# Cột chính: Retrieval
st.header("💬 Truy vấn thông tin")

if retrieval_pipeline is None:
    st.warning("Pipeline truy vấn chưa sẵn sàng. Vui lòng kiểm tra lại cấu hình và kết nối.")
else:
    # Sử dụng session state để lưu trữ lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị các tin nhắn đã có
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Nhận input từ người dùng
    if prompt := st.chat_input("Nhập câu hỏi của bạn ở đây..."):
        # Thêm câu hỏi vào lịch sử chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Xử lý và hiển thị câu trả lời
        with st.chat_message("assistant"):
            response_container = st.empty()
            with st.spinner("Đang tìm kiếm và tạo câu trả lời từ Bedrock..."):
                # Gọi pipeline retrieval để lấy câu trả lời cuối cùng
                response = retrieval_pipeline.search(query=prompt, k=5, strategy='hierarchical')

                # Hiển thị câu trả lời từ LLM
                response_container.markdown(response['answer'])

                # Hiển thị các nguồn đã sử dụng trong expander
                if response['sources']:
                    with st.expander("Xem các nguồn tham khảo"):
                        for i, source in enumerate(response['sources']):
                            st.write(f"**Nguồn {i+1} (Score: {source['final_score']:.4f})**")
                            st.info(source['content'])
        
        # Thêm câu trả lời vào lịch sử chat
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})