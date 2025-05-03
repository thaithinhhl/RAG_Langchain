from typing import Union
import sys
import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag.file_loader import load_pdf_files

def build_vector_db(documents, embedding=None):
    if embedding is None:
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    db = FAISS.from_documents(documents, embedding)
    return db
     
def get_retriever(vector_db, search_type = 'similarity', search_kwargs: dict = {'k': 10}):
    retriever = vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs) 
    return retriever

