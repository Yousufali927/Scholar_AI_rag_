import os
import openai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_DIR = "your_vector_dir"  # replace with actual path
openai.api_key = os.getenv("OPENAI_API_KEY")

def run_query(query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""Use the following context to answer the question.
If you don't know the answer, say "I don't know."

Context:
{context}

Question: {query}
Answer:"""

    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0,
            max_tokens=500,
        )
        return {"answer": response.choices[0].text.strip()}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}








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
VECTOR_DIR = "vectorstore/faiss_index"
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
