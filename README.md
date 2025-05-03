# RAG Langchain System

Hệ thống truy vấn tài liệu thông minh sử dụng LangChain, FastAPI và LangServe.

## Giới thiệu

Hệ thống RAG (Retrieval-Augmented Generation) này cho phép người dùng đặt câu hỏi về tài liệu và nhận câu trả lời thông minh. Hệ thống sử dụng:

- **LangChain**: Framework để phát triển các ứng dụng AI tổng hợp
- **FastAPI**: Web framework hiệu suất cao
- **LangServe**: Triển khai chuỗi LangChain dưới dạng API

## Cài đặt

### Yêu cầu

- Python 3.9+
- Thư viện phụ thuộc trong requirements.txt

### Cài đặt các phụ thuộc

```bash
pip install -r requirements.txt
```

### Cấu trúc thư mục

```
.
├── data_source/
│   └── generative_ai/  # Thư mục chứa tài liệu PDF
├── src/
│   ├── app.py          # API FastAPI
│   ├── base/
│   │   └── llm_model.py   # Mô hình LLM
│   └── rag/
│       ├── app.py      # Ứng dụng RAG
│       ├── file_loader.py # Bộ nạp tài liệu
│       ├── main.py     # Xây dựng RAG Chain
│       ├── offline_rag.py # RAG offline
│       └── vectostore.py # Quản lý vector store
└── README.md
```

## Sử dụng

### Khởi động API

```bash
cd rag_langchain
python src/app.py
```

## Truy cập

Sau khi triển khai:

- **API**: http://localhost:5000
  - API Docs: http://localhost:5000/docs
  - LangServe Playground: http://localhost:5000/langserve/rag/playground/

## API Endpoints

- `GET /`: Kiểm tra API hoạt động
- `GET /check`: Kiểm tra trạng thái API và mô hình
- `POST /api/rag`: Endpoint chính để truy vấn RAG với JSON input
- `GET /query?question=<câu hỏi>`: Truy vấn trực tiếp qua URL parameter
- `GET /langserve/rag/playground/`: LangServe playground

## Khắc phục sự cố

### Lỗi kết nối API

Nếu gặp lỗi "Cannot connect to API" hoặc "Connection refused" như sau:
```
❌ Không thể kết nối tới API: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded with url: /api/check (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f60a96e06a0>: Failed to establish a new connection: [Errno 111] Connection refused'))
```

Đây là lỗi khi client không thể kết nối tới API FastAPI. Hãy thử các giải pháp sau:

1. **Kiểm tra API đã chạy chưa**:
   ```bash
   # Kiểm tra API có đang chạy không
   lsof -i :5000
   # Nếu không có kết quả, API không chạy. Khởi động lại API:
   uvicorn src.app:app --host 0.0.0.0 --port 5000
   ```

2. **Kiểm tra xung đột cổng**: 
   Nếu port 5000 đã bị sử dụng bởi ứng dụng khác, hãy thử dùng port khác:
   ```bash
   uvicorn src.app:app --host 0.0.0.0 --port 8080
   ```

3. **Kiểm tra tường lửa**:
   ```bash
   sudo ufw status   # Ubuntu/Debian
   sudo firewall-cmd --state   # CentOS/RHEL
   ```

### Kiểm tra API với curl

Sử dụng script `test_api.sh` để kiểm tra API:

```bash
./test_api.sh
```

### Cấu hình cho môi trường sản xuất

Để triển khai trong môi trường sản xuất, nên sử dụng:

```bash
# Cài đặt các phụ thuộc bổ sung
pip install gunicorn

# Khởi động API với Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5000 src.app:app
```

## Biến môi trường

Hệ thống sử dụng các biến môi trường sau:

- `PORT`: Port cho API FastAPI (mặc định: 5000)
- `HUGGINGFACE_TOKEN`: Token Hugging Face để tải mô hình (tùy chọn)

## Lưu ý

- Mặc định hệ thống sẽ tìm các mô hình LLM từ Hugging Face. Nếu không có kết nối, hệ thống sẽ sử dụng mô hình GPT-2 cục bộ.
- Hệ thống tìm kiếm token Hugging Face từ biến môi trường HUGGINGFACE_TOKEN.
- Hệ thống sử dụng tài liệu trong thư mục `data_source/generative_ai/`. 
