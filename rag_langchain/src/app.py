import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from langserve import add_routes
import multiprocessing

from base.llm_model import get_hf_llm
from rag.main import build_rag_chain, InputQA, OutputQA

# Khởi tạo LLM
llm = get_hf_llm(temperature=0.9)
genai_docs = "./data_source/generative_ai"

# Khởi tạo chain
def init_chain():
    print("Initializing RAG chain...")
    chain = build_rag_chain(
        llm,
        data_dir=genai_docs,
        data_type="pdf",
        use_multiprocessing=False  
    )
    print("RAG chain initialized successfully!")
    return chain

# Khởi tạo chain toàn cục
genai_chain = None

# ---------------- App - FastAPI ----------------
app = FastAPI(
    title="LangChain RAG Server",
    version="1.0",
    description="API server sử dụng LangChain RAG để truy vấn tài liệu"
)

# ---------------- Middleware ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Khởi tạo chain khi startup
@app.on_event("startup")
async def startup_event():
    global genai_chain
    if genai_chain is None:
        genai_chain = init_chain()
        try:
            add_routes(
                app,
                genai_chain,
                path="/api/generative_ai",
                input_type=InputQA,
                output_type=OutputQA,
            )
            print("LangServe routes added successfully!")
        except Exception as e:
            print(f"Error adding LangServe routes: {e}")

# ---------------- Routes - FastAPI ----------------
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    try:
        global genai_chain
        if genai_chain is None:
            genai_chain = init_chain()
        
        original_question = inputs.question
        print(f"[DEBUG] Câu hỏi gốc nhận được: {original_question}")
        
        result = genai_chain.invoke(inputs)
        
        return {"answer": result}
    except Exception as e:
        print(f"Error in generative_ai endpoint: {e}")
        return {"answer": f"Lỗi: {str(e)}"}

@app.post("/api/raw_question")
async def raw_question(inputs: dict):
    print(f"[DEBUG] Raw question received: {inputs}")
    return {"received_question": inputs.get("question", "")}

@app.get("/playground")
async def playground():
    return RedirectResponse(url="/api/generative_ai/playground")

# ---------------- Main - Run Uvicorn ----------------
if __name__ == "__main__":
    genai_chain = init_chain()
    
    # LangServe routes
    try:
        add_routes(
            app,
            genai_chain,
            path="/api/generative_ai",
            input_type=InputQA,
            output_type=OutputQA,
        )
        print("LangServe routes added successfully!")
    except Exception as e:
        print(f"Error adding LangServe routes: {e}")
    
    multiprocessing.freeze_support()
    
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

