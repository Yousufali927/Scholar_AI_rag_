# backend/ingest_youtube.py

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os

VECTOR_DIR = "vectorstore/faiss_index"

def get_video_id(url: str):
    import re
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

async def process_youtube(youtube_url: str):
    video_id = get_video_id(youtube_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = "\n".join([entry["text"] for entry in transcript_list])

    document = Document(page_content=full_text, metadata={"source": youtube_url})
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([document])

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DIR)

    return {"status": "YouTube transcript processed", "chunks": len(chunks)}
