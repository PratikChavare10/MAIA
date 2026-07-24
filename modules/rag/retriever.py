"""
modules/rag/retriever.py
━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD:
1. ingest.py run केल्यावर vectorstore automatically load होतो
2. .env मध्ये OPENAI_API_KEY असणे आवश्यक आहे

HOW TO USE:
   from modules.rag.retriever import rag_search
   result = rag_search("PM Kisan scheme eligibility")
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from config import VECTORSTORE_PATH, OPENAI_API_KEY
import os

# ── Load Vector Store (once at startup) ──────────
_vectorstore = None
_embeddings  = None

def _load():
    global _vectorstore, _embeddings
    if _vectorstore is None:
        print("Loading RAG vector store...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/LaBSE"
        )
        _vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ RAG Knowledge Base loaded!")

# ── OpenAI Client ─────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

# ── Search Function ───────────────────────────────
def rag_search(query: str) -> dict:
    """
    Documents मधून relevant माहिती शोधतो
    आणि GPT-4 ने answer generate करतो

    Input:
        query (str) → farmer's question in English

    Output:
        dict → {answer}
    """
    _load()

    # Retrieve top 3 relevant chunks
    docs    = _vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Generate answer using GPT-4
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert agricultural advisor for Indian farmers. "
                    "Answer based on the provided context only. "
                    "Be clear, practical, and concise. "
                    "If information is not in context, say so honestly."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context from agricultural documents:\n{context}\n\n"
                    f"Farmer's Question: {query}\n\n"
                    f"Provide a helpful, practical answer:"
                )
            }
        ],
        max_tokens=400,
        temperature=0.3
    )

    return {"answer": response.choices[0].message.content}
