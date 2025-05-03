# RAG Langchain System

Hệ thống truy vấn tài liệu thông minh sử dụng LangChain, FastAPI, LangServe và Gradio.

## Giới thiệu

Hệ thống RAG (Retrieval-Augmented Generation) này cho phép người dùng đặt câu hỏi về tài liệu và nhận câu trả lời thông minh. Hệ thống sử dụng:

- **LangChain**: Framework để phát triển các ứng dụng AI tổng hợp
- **FastAPI**: Web framework hiệu suất cao
- **LangServe**: Triển khai chuỗi LangChain dưới dạng API
- **Gradio**: Giao diện người dùng thân thiện

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
│   ├── gradio_ui.py    # Giao diện Gradio
│   ├── base/
│   │   └── llm_model.py   # Mô hình LLM
│   └── rag/
│       └── main.py     # Xây dựng RAG Chain
├── deploy.py          # Script triển khai
└── README.md
```

## Sử dụng

### Triển khai toàn bộ hệ thống (API và UI)

```bash
python deploy.py
```

### Triển khai chỉ API

```bash
python deploy.py --mode api
```

### Triển khai chỉ UI

```bash
python deploy.py --mode ui
```

### Sử dụng với port tùy chỉnh

```bash
python deploy.py --api-port 5000 --ui-port 7860
```

## Truy cập

Sau khi triển khai:

- **API**: http://localhost:5000
  - API Docs: http://localhost:5000/docs
  - LangServe Playground: http://localhost:5000/langserve/rag/playground/
- **Gradio UI**: http://localhost:7860

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

Đây là lỗi khi Gradio UI không thể kết nối tới API FastAPI. Hãy thử các giải pháp sau:

1. **Kiểm tra API đã chạy chưa**:
   ```bash
   # Kiểm tra API có đang chạy không
   lsof -i :8000
   # Nếu không có kết quả, API không chạy. Khởi động lại API:
   python deploy.py --mode api --api-port 8000 --debug
   ```

2. **Kiểm tra xung đột cổng**: 
   Nếu port 8000 đã bị sử dụng bởi ứng dụng khác, hãy thử dùng port khác:
   ```bash
   python deploy.py --api-port 8080 --ui-port 7070
   ```

3. **Kiểm tra URL API trong Gradio UI**:
   Hãy chắc chắn URL API đã được cấu hình đúng trong biến môi trường:
   ```bash
   export API_URL="http://localhost:8000/api/rag"
   python src/gradio_ui.py
   ```

4. **Kiểm tra tường lửa**:
   ```bash
   sudo ufw status   # Ubuntu/Debian
   sudo firewall-cmd --state   # CentOS/RHEL
   ```

5. **Chạy ở chế độ debug**:
   ```bash
   python deploy.py --debug
   ```

### Các lỗi khác

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
- `GRADIO_SERVER_PORT`: Port cho Gradio UI (mặc định: 7860)
- `API_URL`: URL của API để Gradio UI kết nối (mặc định: http://localhost:5000/api/rag)
- `HUGGINGFACE_TOKEN`: Token Hugging Face để tải mô hình (tùy chọn)

## Lưu ý

- Mặc định hệ thống sẽ tìm các mô hình LLM từ Hugging Face. Nếu không có kết nối, hệ thống sẽ sử dụng mô hình GPT-2 cục bộ.
- Hệ thống tìm kiếm token Hugging Face từ biến môi trường HUGGINGFACE_TOKEN hoặc trong file token.
- Hệ thống sử dụng tài liệu trong thư mục `data_source/generative_ai/`. 
