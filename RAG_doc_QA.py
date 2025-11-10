import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.Chains.combine_documents import create_stuff_documents_chain
from langchain.Chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

groq_api_key=os.getenv('GROQ_API_KEY')
llm=ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

prompt=ChatPromptTemplate.from_template(

    """
    Answer the question based on the provided context only 
    Please provide the most accurate response based on question 
    <context>
    {context}   
    <context>

    Question:{input}

"""
)
import tempfile
@st.cache_data(show_spinner="Processing PDF...")
def create_vectorstore_from_pdfs(uploaded_files):
    
    docs = []

    try:
        for uploaded_file in uploaded_files:
                print(uploaded_file)
                temppdf=f'./temp.pdf'
                with open(temppdf,'wb') as file:
                    file.write(uploaded_files.getvalue())
                    file_name=uploaded_files.name 
                    print('written')  
                loader=PyPDFLoader(temppdf)
                docs=loader.load()
                docs.extend(docs)
    except Exception as e_loader:
                # Loading failed (corrupted / not a PDF / encrypted, etc.)
                reason = f"Loading error: {e_loader}"
                st.warning(f"⚠️ Skipping '{uploaded_file.name}': {reason}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader('research')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vector=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    return 1

import time 


def create_stuff_documents_chain(llm, prompt):
    """Simple replacement for LangChain's create_stuff_documents_chain"""
    # from langchain_core.documents import Document

    def run(docs, input_text):
        context = "\n\n".join(doc.page_content for doc in docs)
        full_input = prompt.format(context=context, input=input_text)
        return llm.invoke(full_input)

    return run
class RetrievalChain:
    """
    Custom RetrievalChain that mimics LangChain's built-in behavior.
    """
    def __init__(self, retriever, qa_chain):
        self.retriever = retriever
        self.qa_chain = qa_chain

    def invoke(self, inputs):
        # Support both dict and string inputs
        user_input = inputs["input"] if isinstance(inputs, dict) else inputs

        # Retrieve relevant documents
        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(user_input)
        else:
            docs = self.retriever.get_relevant_documents(user_input)

        # Run QA chain on retrieved docs
        return self.qa_chain(docs, user_input)


def create_retrieval_chain(retriever, qa_chain):
    """
    Factory function returning an instance with .invoke() method.
    """
    return RetrievalChain(retriever, qa_chain)

st.title('PDF RAG Document Q&A search')

uploaded_files= st.file_uploader('choose pdf ',accept_multiple_files=False)

if uploaded_files:
    st.write("⏳ Loading document...")
    st.session_state.vector=create_vectorstore_from_pdfs(uploaded_files)
    st.success("Ready to go!")
    
    user_prompt=st.text_input("Enter your query")
    if user_prompt:
    
        documment_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, documment_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        print(f"Response time = {time.process_time() - start:.2f}s")
        print(type(response))

        st.write(response.content)

        with st.expander('document similarity'):
            docs = retriever.invoke(user_prompt)
            for i, doc in enumerate(docs):
                st.write(doc.page_content)
                st.write('--------------')
