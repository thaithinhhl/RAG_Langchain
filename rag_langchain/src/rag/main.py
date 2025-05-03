from typing import Dict, Any, List, Union
try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

from rag.file_loader import load_pdf_from_dir
from rag.vectostore import build_vector_db, get_retriever
from rag.offline_rag import create_rag_promp
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")

def _extract_query(query_input: Union[str, dict, BaseModel]) -> str:
    if isinstance(query_input, str):
        return query_input
    elif isinstance(query_input, dict) and "question" in query_input:
        return query_input["question"]
    elif hasattr(query_input, "question"):
        return query_input.question
    return ""

def _safe_retrieval(retriever, query_input: Union[str, dict, BaseModel]) -> List[Document]:
    query = _extract_query(query_input)
    if not query:
        return []
    return retriever.invoke(query)

def build_rag_chain(llm, data_dir: str, data_type: str, use_multiprocessing: bool = True) -> Any:
    workers = 2 if use_multiprocessing else 1
    
    documents = load_pdf_from_dir(data_dir, workers=workers)
    
    vector_db = build_vector_db(documents)
    retriever = get_retriever(vector_db)
    
    def safe_retriever(x):
        if isinstance(x, str):
            return retriever.invoke(x)
        elif isinstance(x, dict) and "question" in x:
            return retriever.invoke(x["question"])
        elif hasattr(x, "question"):
            return retriever.invoke(x.question)
        else:
            print(f"Không nhận diện được định dạng đầu vào: {type(x)}")
            return []
    
    def extract_question(x):
        if isinstance(x, str):
            return x
        elif isinstance(x, dict) and "question" in x:
            return x["question"]
        elif hasattr(x, "question"):
            return x.question
        else:
            return "Không tìm thấy câu hỏi"
    
    retriever_chain = RunnableParallel({
        "context": safe_retriever,
        "question": extract_question
    })
    
    rag_chain = retriever_chain | create_rag_promp(retriever, llm)
    
    return rag_chain