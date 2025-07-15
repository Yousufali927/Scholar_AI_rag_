import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize embedding model and vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "vectorstore/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_query_rag(user_query: str) -> dict:
    """
    Runs a RAG-style query using FAISS vector store for context retrieval
    and OpenAI Chat Completion for final answer generation.
    """
    # Step 1: Retrieve relevant documents
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    docs = retriever.get_relevant_documents(user_query)

    # Step 2: Build RAG context from top documents
    rag_context = "\n\n".join([doc.page_content for doc in docs])

    # Step 3: Construct Chat Prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent assistant with access to relevant documents. "
                "Use the provided context to answer the user's question accurately and concisely."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{rag_context}\n\n"
                f"Question: {user_query}\n\n"
                "Answer using only the information from the context above."
            )
        }
    ]

    # Step 4: Get response from OpenAI Chat Completion
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [doc.metadata.get("source", "N/A") for doc in docs]
    }
