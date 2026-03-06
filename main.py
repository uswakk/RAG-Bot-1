#from dotenv import load_dotenv

import streamlit as st
#from langchain_anthropic import ChatAnthropic
#from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama


from langchain_ollama import OllamaEmbeddings
#from langchain_google_genai import GoogleGenerativeAIEmbeddings

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

import time
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader

#to disable the file watcher for streamlit
import os 
import torch 
torch.classes.__path__ = []

#load_dotenv()

#make functions to make everything simpler

def get_raw_text_from_pdfs(files):
    raw_text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            raw_text += page.extract_text()
    return raw_text

def split_text(text, chunk_size=800, chunk_overlap=150):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,  # chunk size (characters)
    chunk_overlap=chunk_overlap,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
    )

    chunks = text_splitter.split_text(text)

    return chunks

def create_vector_store(chunks):
       
     

        embeddings = OllamaEmbeddings(model="granite-embedding")

        embedding_dim = len(embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        document_ids = vector_store.add_texts(texts=chunks)
        return document_ids, vector_store



model = ChatOllama(
    model="tinyllama",
    temperature=0,
    # other params...
)

st.title("Ask Questions Related To Your Documents")
st.header("Upload your documents to get started")

files = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"], accept_multiple_files=True)
button_state = st.button("Process Document")  

if button_state:
    st.write("Processing your document...")

   
    if files is not None:    
        #time.sleep(2)  # Simulate processing time

        #first get raw text from the document
        raw_text = get_raw_text_from_pdfs(files)
        #st.write("Text has been split into the following chunks:")
        chunks = split_text(raw_text)
        #st.write(chunks)
        document_ids, vector_store = create_vector_store(chunks)
        #indexing 
        st.session_state.vector_store = vector_store


    
user_query = st.text_input("Write Your Query:")

if user_query:
    st.write("Processing your query...")
    #from langchain.agents import create_agent


    retrieved_docs = st.session_state.vector_store.similarity_search(user_query, k=1)
    # If desired, specify custom instructions
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    Use the following context to answer the question. Make the answer precise. If the answer is not contained within the context, say you don't know.

    Context:
    {context}

    Question:
    {user_query}
    """

    ai_msg = model.invoke(prompt)
    st.write(ai_msg.content)
