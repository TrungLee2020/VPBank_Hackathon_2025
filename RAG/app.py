# --- app.py ---

import streamlit as st
import boto3
import os
import time
import asyncio
from dotenv import load_dotenv
from pathlib import Path

try:
    from unstructured.partition.auto import partition
    from unstructured.staging.base import convert_to_markdown
except ImportError:
    st.error("Thư viện 'unstructured' chưa được cài đặt. Vui lòng chạy: pip install \"unstructured[docx,xlsx,csv]\"")
    partition = None

try:
    from llama_parse import LlamaParse
except ImportError:
    st.error("Thư viện 'llama-parse' chưa được cài đặt. Vui lòng chạy: pip install llama-parse")
    LlamaParse = None

# Import các pipeline
from indexing_pipeline import IndexingPipeline
from retrieval_pipeline import RetrievalPipeline
from config import RAGConfig

# Tải biến môi trường từ file .env
load_dotenv()

# --- Cấu hình và Khởi tạo ---
st.set_page_config(page_title="Vietnamese RAG System", layout="wide")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")
LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

s3_client = boto3.client('s3', aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), region_name=AWS_REGION)
rag_config = RAGConfig()
rag_config.vector_store_type = 'opensearch'
rag_config.reranker.enabled = True 

@st.cache_resource
def get_retrieval_pipeline():
    try:
        return RetrievalPipeline(config=rag_config)
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo Retrieval Pipeline: {e}")
        return None

retrieval_pipeline = get_retrieval_pipeline()

# --- Parsers & S3 ---

def upload_to_s3(file_path_or_obj, bucket, object_name):
    """Tải file lên S3."""
    try:
        if isinstance(file_path_or_obj, str) or isinstance(file_path_or_obj, Path):
            s3_client.upload_file(str(file_path_or_obj), bucket, object_name)
        else:
            s3_client.upload_fileobj(file_path_or_obj, bucket, object_name)
        return True
    except Exception as e:
        st.error(f"Lỗi khi tải file lên S3: {e}")
        return False

async def parse_with_llamaparse(file_path: str) -> str | None:
    """Sử dụng LlamaParse cho PDF và ảnh (các tác vụ cần OCR)."""
    st.info("Sử dụng LlamaParse (cho PDF/ảnh)...")
    if not LlamaParse or not LLAMA_API_KEY:
        st.error("LlamaParse hoặc API key chưa được cấu hình.")
        return None
    parser = LlamaParse(
        api_key=LLAMA_API_KEY, 
        result_type="markdown", 
        num_workers=4,
        verbose=True, 
        language="vi",
        )
    try:
        documents = await parser.aload_data(file_path=file_path)
        return documents[0].text if documents else None
    except Exception as e:
        st.error(f"Lỗi với LlamaParse: {e}")
        return None

def parse_with_unstructured(file_path: str) -> str | None:
    """Sử dụng thư viện Unstructured cho các file office (DOCX, XLSX, CSV)."""
    st.info("Sử dụng Unstructured.io để trích xuất...")
    if not partition:
        st.error("Thư viện Unstructured chưa được khởi tạo.")
        return None
    try:
        # Hàm partition tự động nhận diện và xử lý file
        elements = partition(filename=file_path)
        # Chuyển đổi các elements (Title, Table, NarrativeText, etc.) thành một chuỗi markdown duy nhất
        return convert_to_markdown(elements)
    except Exception as e:
        st.error(f"Lỗi với Unstructured: {e}")
        return None

# ==========================================================
# Bôr sung hàm xử lý reformat_markdown để upload lên s3
# ==========================================================
async def reformat_markdown(file_path: str):
    """
    Tạo lại định dạng đầu mục heading cho văn bản
    """
    pass

async def dispatch_file_parser(file_path: str) -> Path | None:
    """
    Điều phối file đến đúng trình phân tích dựa trên đuôi file.
    Trả về đường dẫn đến file markdown đã được tạo.
    """
    file_extension = Path(file_path).suffix.lower()
    markdown_text = None

    # Lựa chọn parser phù hợp
    if file_extension in ['.pdf', '.jpg', '.jpeg', '.png']:
        markdown_text = await parse_with_llamaparse(file_path)
    elif file_extension in ['.docx', '.xlsx', '.xls', '.csv']:
        # Unstructured xử lý tốt các định dạng này
        markdown_text = parse_with_unstructured(file_path)
    else:
        st.warning(f"Không hỗ trợ định dạng file '{file_extension}'.")
        return None
    
    # Lưu kết quả markdown vào file tạm và trả về đường dẫn
    if markdown_text:
        temp_dir = Path("./temp_markdowns")
        temp_dir.mkdir(exist_ok=True)
        output_filename = f"{Path(file_path).stem}.md"
        markdown_path = temp_dir / output_filename
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        st.success("Trích xuất nội dung thành công!")
        return markdown_path
    else:
        st.error("Trích xuất nội dung thất bại, không có văn bản được trả về.")
        return None

async def run_indexing(markdown_path: Path):
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

# --- Giao diện Streamlit ---
st.title("Hệ thống RAG Tiếng Việt (Unstructured + LlamaParse)")
st.markdown("---")

with st.sidebar:
    st.header("📚 Quản lý tài liệu")
    st.subheader("1. Tải lên tài liệu mới")
    
    uploaded_file = st.file_uploader(
        "Chọn file (PDF, DOCX, XLSX, CSV, PNG, JPG...)", 
        type=['pdf', 'docx', 'xlsx', 'xls', 'csv', 'png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        st.write(f"Đã chọn: `{uploaded_file.name}`")
        if st.button("Bắt đầu xử lý"):
            with st.status("**Đang xử lý file...**", expanded=True) as status:
                
                # Bước 1: Upload file gốc lên S3
                s3_original_path = f"uploads/{uploaded_file.name}"
                st.write(f"Bước 1: Tải file gốc lên S3 tại: `{s3_original_path}`...")
                upload_success = upload_to_s3(uploaded_file, S3_BUCKET_NAME, s3_original_path)
                if not upload_success:
                    status.update(label="Tải file gốc lên thất bại!", state="error", expanded=True)
                    st.stop()
                st.write("   => Tải lên S3 thành công.")
                
                # Bước 2: Lưu file tạm để parser đọc
                temp_upload_dir = Path("./temp_uploads")
                temp_upload_dir.mkdir(exist_ok=True)
                temp_file_path = temp_upload_dir / uploaded_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Bước 3: Điều phối file và tạo markdown
                st.write("Bước 2: Trích xuất nội dung sang Markdown...")
                local_markdown_path = asyncio.run(dispatch_file_parser(str(temp_file_path)))
                
                # Xóa file upload tạm
                os.remove(temp_file_path)

                # Nếu có markdown, upload lên S3 và tiến hành indexing
                if local_markdown_path:
                    st.write(f"   => File Markdown được tạo tại: `{local_markdown_path}`")
                    
                    # Bước 4: Upload file markdown đã xử lý lên S3
                    s3_markdown_path = f"processed-markdowns/{local_markdown_path.name}"
                    st.write(f"Bước 3: Tải file markdown lên S3 tại: `{s3_markdown_path}`...")
                    upload_md_success = upload_to_s3(local_markdown_path, S3_BUCKET_NAME, s3_markdown_path)
                    if upload_md_success:
                        st.write("   => Tải lên S3 thành công.")
                    
                    # Bước 5: Chạy pipeline indexing
                    st.write("Bước 4: Lập chỉ mục tài liệu...")
                    asyncio.run(run_indexing(local_markdown_path))
                    
                    status.update(label="Hoàn tất xử lý!", state="complete", expanded=True)
                    st.balloons()
                    
                    # Xóa file markdown tạm trên máy local sau khi đã xong việc
                    os.remove(local_markdown_path)
                else:
                    status.update(label="Trích xuất nội dung thất bại!", state="error", expanded=True)


# --- Cột chính: Retrieval ---
st.header("💬 Truy vấn thông tin")
if retrieval_pipeline is None:
    st.warning("Pipeline truy vấn chưa sẵn sàng.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Nhập câu hỏi của bạn ở đây..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_container = st.empty()
            with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
                response = retrieval_pipeline.search(query=prompt, k=5, strategy='hierarchical')
                response_container.markdown(response['answer'])
                if response['sources']:
                    with st.expander("Xem các nguồn tham khảo"):
                        for i, source in enumerate(response['sources']):
                            st.write(f"**Nguồn {i+1} (Score: {source['final_score']:.4f})**")
                            st.info(source['content'])
        
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})