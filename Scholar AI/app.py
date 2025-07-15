# app.py
# front end streamlit app

import streamlit as st
import requests

BACKEND = "http://127.0.0.1:8000"

st.set_page_config(page_title="ScholarAI", layout="wide")
st.title("📚 ScholarAI — RAG Research Assistant")

# Sidebar file upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Uploading and processing PDF..."):
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        res = requests.post(f"{BACKEND}/upload_pdf", files=files)
        st.sidebar.success("✅ PDF processed")
        st.sidebar.write(res.json())

# YouTube Link
st.sidebar.header("Or Enter YouTube URL")
youtube_url = st.sidebar.text_input("Paste YouTube link")

if st.sidebar.button("Process YouTube"):
    with st.spinner("Fetching transcript and embedding..."):
        res = requests.post(f"{BACKEND}/upload_youtube", data={"youtube_url": youtube_url})
        if res.status_code == 200:
            st.sidebar.success("✅ YouTube processed")
            st.sidebar.write(res.json())
        else:
            st.sidebar.error("Failed to process video")

# Main Query Input
st.subheader("Ask a question based on the uploaded content")
query = st.text_input("Your question here:")

if st.button("Get Answer"):
    with st.spinner("Thinking..."):
        res = requests.post(f"{BACKEND}/query", data={"query": query})
        if res.status_code == 200:
            result = res.json()
            st.write("### ✅ Answer:")
            st.write(result["answer"])
            if result.get("sources"):
                st.write("### 📌 Sources:")
                st.write(result["sources"])
        else:
            st.error("Something went wrong querying the backend.")
