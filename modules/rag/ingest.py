"""
modules/rag/ingest.py
━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD BEFORE RUNNING:
1. PDFs गोळा करा आणि data/documents/ मध्ये ठेवा:
   - pm_kisan_scheme.pdf     → pmkisan.gov.in
   - pmfby_insurance.pdf     → pmfby.gov.in
   - crop_disease_manual.pdf → icar.org.in
   - fertilizer_guide.pdf    → kvk.icar.gov.in
   - pest_control.pdf        → (Google वर search करा)

HOW TO RUN: (फक्त एकदा run करा — documents change झाल्यावर परत)
   python modules/rag/ingest.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import DOCUMENTS_PATH, VECTORSTORE_PATH

def ingest_documents():
    """
    PDFs load करून FAISS vector store बनवतो
    """
    # ── Load all PDFs ────────────────────────────
    all_docs = []
    pdf_files = [f for f in os.listdir(DOCUMENTS_PATH)
                 if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️  No PDFs found in {DOCUMENTS_PATH}")
        print("   Add PDF files and run again.")
        return

    for pdf in pdf_files:
        path = os.path.join(DOCUMENTS_PATH, pdf)
        print(f"Loading: {pdf}")
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    print(f"\n✅ Loaded {len(all_docs)} pages from {len(pdf_files)} PDFs")

    # ── Split into chunks ────────────────────────
    # ADD: chunk_size adjust करा जर answers too short/long येत असतील
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)
    print(f"✅ Created {len(chunks)} chunks")

    # ── Create Embeddings ────────────────────────
    # LaBSE → multilingual (Marathi/Hindi/English support)
    print("\nCreating embeddings (this takes a few minutes)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/LaBSE"
    )

    # ── Save to FAISS ────────────────────────────
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"\n✅ Vector store saved: {VECTORSTORE_PATH}")
    print("✅ RAG Knowledge Base ready!")

if __name__ == "__main__":
    ingest_documents()
