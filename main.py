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
from docx import Document

#to disable the file watcher for streamlit
import os 
import torch 
torch.classes.__path__ = []

#load_dotenv()

#make functions to make everything simpler

def get_text_from_docx(file):
    doc = Document(file)
    full_text = []
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def get_raw_text_from_pdfs(file):
    raw_text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        raw_text += page.extract_text()
    return raw_text

def get_text_from_files(files):
    raw_text = ""
    for file in files:
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            text = get_raw_text_from_pdfs(file)
        elif ext == "docx":
            text = get_text_from_docx(file)
        elif ext == "txt":
            text = file.read().decode("utf-8")
        raw_text += text + "\n"
    return raw_text

def split_text(text, chunk_size=1500, chunk_overlap=150):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,  # chunk size (characters)
    chunk_overlap=chunk_overlap,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
    )

    chunks = text_splitter.split_text(text)

    return chunks

#add caching to the vector store creation to speed up subsequent queries
@st.cache_resource
def create_vector_store(chunks):
       
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

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
    model="qwen:1.8b",
    temperature=0.1, # Small increase allows for slight variation
    num_predict=250, # Limits the total output length to prevent infinite loops
    model_kwargs={
        "repeat_penalty": 1.2, # Higher value (1.1 - 1.5) prevents repetition
        "top_p": 0.9,          # Focuses the model on the most likely words
    }
)

st.title("Ask Questions Related To Your Documents")
st.header("Upload your documents to get started")

files = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"], accept_multiple_files=True)
button_state = st.button("Process Document")  

if button_state:
    with st.spinner("Processing..."):
   
        if files is not None:    
            #time.sleep(2)  # Simulate processing time

            #first get raw text from the document
            raw_text = get_text_from_files(files)
            #st.write("Text has been split into the following chunks:")
            chunks = split_text(raw_text)
            #st.write(chunks)
            document_ids, vector_store = create_vector_store(chunks)
            #indexing 
            st.session_state.vector_store = vector_store



user_query = st.text_input("Write Your Query:")

if user_query:

    if "vector_store" not in st.session_state:
        st.warning("Please upload and process a document first.")
    else:
        st.write("Processing your query...")
        #from langchain.agents import create_agent


        retrieved_docs = st.session_state.vector_store.similarity_search(user_query, k=4)
        # If desired, specify custom instructions
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
        ### INSTRUCTION
        You are a helpful assistant. Use the provided Context to answer the Question.
        - If the Question asks for a summary, provide 3-5 concise bullet points.
        - If the Question is a specific query, provide a direct and precise answer.
        - If the answer is not in the Context, simply state: "I do not have enough information."
        - DO NOT repeat the same sentence or phrase multiple times.

        ### CONTEXT
        {context}

        ### QUESTION
        {user_query}

        ### ANSWER (concise and direct):
        """

        ai_msg = model.invoke(prompt)
        st.write(ai_msg.content)
