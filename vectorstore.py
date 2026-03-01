import fitz  # PyMuPDF
import tempfile
import streamlit as st
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_data(show_spinner="Processing PDF...")
def create_fast_vectorstore(uploaded_file):
    if uploaded_file is None:
        st.error("Upload a PDF")
        return None

    progress = st.progress(0, "Saving PDF...")

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:        #  Save PDF temporarily
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    
    pdf = fitz.open(temp_path)            #  Open using PyMuPDF (fastest)
    total_pages = pdf.page_count

    docs = []
    progress.progress(5, f"Reading pages: 0 / {total_pages}")

   
    for i in range(total_pages):             #  Extract text super fast
        page = pdf.load_page(i)
        text = page.get_text("text")

        docs.append(Document(page_content=text, metadata={"page": i + 1})
        )
        
        pct = int((i + 1) / total_pages * 60)  # use 60% for loading
        progress.progress(pct, f"Reading pages: {i + 1} / {total_pages}")

    pdf.close()

    progress.progress(70, "Chunking text...")

    splitter = RecursiveCharacterTextSplitter(                    # Chunk using LangChain
        chunk_size=1500,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    progress.progress(85, "thinking...")

    embeddings = HuggingFaceEmbeddings()                        #  Embeddings + FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    progress.progress(100, "Completed!")

    return vectorstore




