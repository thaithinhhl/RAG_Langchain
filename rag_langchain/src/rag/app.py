import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from base.llm_model import get_hf_llm
from rag.main import build_rag_chain, InputQA, OutputQA
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

llm = get_hf_llm(
    model_name=MODEL_NAME,
    temperature=0.7,
    max_new_tokens=512
)

genai_docs = "./data_source/generative_ai"

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

genai_chain = build_rag_chain(
    llm=llm,
    data_dir=genai_docs,
    data_type="pdf",
)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using LangChain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG API đang chạy!"}

@app.get("/check")
async def check():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}

@app.get("/query")
async def query(question: str):
    answer = genai_chain.invoke(question)
    return {"question": question, "answer": answer}

def rag_interface(question):
    response = genai_chain.invoke(question)
    return response

# demo = gr.Interface(
#     fn=rag_interface,
#     inputs=gr.Textbox(lines=2, placeholder="Nhập câu hỏi của bạn..."),
#     outputs=gr.Textbox(label="Câu trả lời"),
#     title="Hệ thống RAG",
#     description="Hãy đặt câu hỏi liên quan đến tài liệu của bạn"
# )

# app = gr.mount_gradio_app(app, demo, path="/gradio")

# --------- Langserve Routes - Playground -------------
add_routes(
    app,
    genai_chain,
    path="/generative_ai"
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

