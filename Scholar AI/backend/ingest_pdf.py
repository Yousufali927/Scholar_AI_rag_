import os
from dotenv import load_dotenv
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
load_dotenv()

DOCUMENTS_DIR = "documents"
BASE_DIR = Path(__file__).resolve().parent.parent 
VECTOR_DIR = BASE_DIR / "vectorstore/faiss_index"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)



async def process_pdf(file: UploadFile):
    pdf_path = os.path.join(DOCUMENTS_DIR, file.filename)
    vectorstore_path = Path(VECTOR_DIR)

    # If the vectorstore already exists, skip processing
    if vectorstore_path.exists() and any(vectorstore_path.iterdir()):
        return {"status": "Vectorstore already exists, skipping."}

    # Otherwise, process PDF
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DIR)

    return {"status": "PDF processed", "chunks": len(chunks)}
















##############

async def process_pdf(file: UploadFile):
    pdf_path = os.path.join(DOCUMENTS_DIR, file.filename)
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DIR)

    return {"status": "PDF processed", "chunks": len(chunks)}
