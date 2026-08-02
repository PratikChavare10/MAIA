

import os
from config import GROQ_API_KEY, VECTORSTORE_PATH
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ── Load Vector Store & LLM (once at startup) ──────────
_vectorstore = None
_embeddings = None
_llm = None


def _load():
    global _vectorstore, _embeddings, _llm
    if _vectorstore is None:
        print("Loading RAG vector store & Groq LLM...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/LaBSE"
        )
        _vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            _embeddings,
            allow_dangerous_deserialization=True,
        )


        _llm = ChatGroq(
            model="llama-3.1-8b-instant", temperature=0.3, max_tokens=400
        )
        print("✅ RAG Knowledge Base and Groq LLM loaded!")


# ── Search Function ───────────────────────────────
def rag_search(query: str) -> dict:

    _load()

    # Retrieve top 3 relevant chunks
    docs = _vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])



    # Messages तयार करणे
    messages = [
        SystemMessage(
            content=(
                "You are an expert agricultural advisor for Indian farmers. "
                "Answer based on the provided context only. "
                "Be clear, practical, and concise. "
                "If information is not in context, say so honestly."
            )
        ),
        HumanMessage(
            content=(
                f"Context from agricultural documents:\n{context}\n\n"
                f"Farmer's Question: {query}\n\n"
                f"Provide a helpful, practical answer:"
            )
        ),
    ]


    # Generate answer using Groq
    response = _llm.invoke(messages)

    return {"answer": response.content}
