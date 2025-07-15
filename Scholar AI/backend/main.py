from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.ingest_pdf import process_pdf
from backend.ingest_youtube import process_youtube
from backend.query_engine import run_query_rag
import os
from dotenv import load_dotenv
import openai
from backend.query_engine import run_query_rag

openai.api_key = os.getenv("OPENAI_API_KEY")

load_dotenv()

app = FastAPI()

# Enable CORS for local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "ScholarAI backend is running"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):
    return await process_pdf(file)

@app.post("/upload_youtube")
async def upload_youtube(youtube_url: str = Form(...)):
    return await process_youtube(youtube_url)

@app.post("/query")
async def query_endpoint(query: str = Form(...)):
    return run_query_rag(query)

@app.post("/query")
async def query_endpoint(payload: dict):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query not provided")

    return run_query_rag(query)