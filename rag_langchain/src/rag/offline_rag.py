import re
from typing import Any, Dict, List, Optional
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def parse(text: str) -> str:
    return extract_answer(text)

def extract_answer(text_response: str) -> str:
    pattern: str = r"Answer:\s*(.*)"
    match = re.search(pattern, text_response, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        return answer_text
    else:
        return text_response

def format_docs(docs):
    if not docs or len(docs) == 0:
        return "Không tìm thấy thông tin liên quan trong tài liệu."
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_promp(retriever, llm):
    # Load prompt từ hub
    prompt = hub.pull("rlm/rag-prompt")
    
    prompt_chain = (
        {
            "context": lambda x: x["context"],  
            "question": lambda x: x["question"]  
        }
        | prompt 
        | llm
        | StrOutputParser()
        | extract_answer
    )
    
    # Store the last question for tracking
    last_question = {"value": ""}
    
    # Define a custom wrapper function to track the original question
    def track_question(x):
        if isinstance(x, str):
            last_question["value"] = x
        elif isinstance(x, dict) and "question" in x:
            last_question["value"] = x["question"]
        elif hasattr(x, "question"):
            last_question["value"] = x.question
        
        result = prompt_chain.invoke(x)
        print(f"Câu hỏi gốc: {last_question['value']}")
        return result
    
    # Wrap original chain with a lambda that preserves the question
    custom_chain = RunnableLambda(track_question)
    
    return custom_chain