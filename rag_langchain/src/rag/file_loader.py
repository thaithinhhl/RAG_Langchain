from typing import Union 
from langchain_community.document_loaders import PyPDFLoader
import multiprocessing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from tqdm import tqdm

def remove_urf_8(text: str) -> str:
    return ''.join(char for char in text if ord(char) < 128)

'''load PDF file'''

def load_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path, extract_images = True)
    docs  = loader.load()
    for doc in docs: 
        doc.page_content = remove_urf_8(doc.page_content)
    return docs 

'''processing cpu'''

def process_cpu():
    return multiprocessing.cpu_count()

'''chia tài liệu'''
def split_documents(documents, split_kwargs:dict) -> List[str]:
    spliter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", ". ", " ", ""],
        chunk_size = split_kwargs.get("chunk_size", 200),
        chunk_overlap = split_kwargs.get("chunk_overlap", 0),
    )
    return spliter.split_documents(documents)

'''xử lý 1 file pdf'''

def process_single_pdf(pdf_and_kwargs):
    pdf_file, split_kwargs = pdf_and_kwargs
    if pdf_file.endswith(".pdf"):
        docs = load_pdf(pdf_file)
        return split_documents(docs, split_kwargs)
    else:
        raise ValueError(f"File {pdf_file} is not a PDF file")
    
''' xu ly nhieu file pdf'''
def load_pdf_files(pdf_files: Union[str, List[str]], workers: int = 1, split_kwargs: dict = None) -> List[str]:
    if split_kwargs is None:
        split_kwargs = {"chunk_size": 300, "chunk_overlap": 0}

    if isinstance(pdf_files, str):
        pdf_files = [pdf_files]

    if len(pdf_files) == 0:
        raise ValueError("No files found")

    num_processes = min(process_cpu(), workers)

    process_args = [(pdf_file, split_kwargs) for pdf_file in pdf_files]

    # Xử lý song song với multiprocessing
    doc_split = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        total_files = len(pdf_files)
        with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
            for result in pool.imap_unordered(process_single_pdf, process_args):
                doc_split.extend(result)
                pbar.update(1)

    return doc_split

'''load pdf tu directory'''
def load_pdf_from_dir(dir_path: str, workers: int = 1, split_kwargs: dict = None) -> List[str]:
    import glob
    files = glob.glob(f"{dir_path}/*.pdf")
    if len(files) == 0:
        raise ValueError(f"No pdf files found in {dir_path}")
    return load_pdf_files(files, workers, split_kwargs)



# if __name__ == "__main__":
#     # Test load_pdf_files
#     pdf_files = ["/home/thinhtran/RAG/Attention Is All You Need.pdf"]
#     split_kwargs = {"chunk_size": 100, "chunk_overlap": 20}
#     result = load_pdf_files(pdf_files, workers=1, split_kwargs=split_kwargs)
#     print("Nội dung đầu tiên của file Attention:")
#     print(result[1].page_content)
