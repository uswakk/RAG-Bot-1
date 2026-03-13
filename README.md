# Mini RAG Bot

## Overview

This project is a **Mini Retrieval-Augmented Generation (RAG) Bot** that allows users to upload their documents and ask questions related to the uploaded content.

The application processes the documents, converts the text into embeddings, stores them in a vector database, and retrieves the most relevant context when a user asks a question. The retrieved information is then used by a language model to generate a precise answer.

## Features

- Upload documents (PDF, DOCX, TXT) and extract text.  
- Accurate PDF parsing with **pdfplumber**.  
- Split documents into smaller, indexed chunks.  
- Generate embeddings for each chunk using **Ollama embeddings**.  
- Store embeddings in a **FAISS vector store** for fast similarity search.  
- Retrieve the most relevant chunks for a query.  
- Generate concise, factual answers using **Qwen 1.8B**.  
- Streamlit-based simple and interactive UI.  
- Latency optimization: context truncation and controlled chunk retrieval. 

# Mini RAG Bot

## Overview

**Mini RAG Bot** is a lightweight Retrieval-Augmented Generation (RAG) application that allows users to upload documents (PDF, DOCX, TXT) and ask questions about their content. The system extracts text, converts it into embeddings, stores them in a FAISS vector database, and retrieves relevant context to provide precise answers using a language model.

This version improves text extraction accuracy using **pdfplumber** for PDFs, optimizes context truncation to reduce latency, and uses **Qwen 1.8B** with **Ollama embeddings** for high-quality answers.

---

## Features

- Upload documents (PDF, DOCX, TXT) and extract text.  
- Accurate PDF parsing with **pdfplumber**.  
- Split documents into smaller, indexed chunks.  
- Generate embeddings for each chunk using **Ollama embeddings**.  
- Store embeddings in a **FAISS vector store** for fast similarity search.  
- Retrieve the most relevant chunks for a query.  
- Generate concise, factual answers using **Qwen 1.8B**.  
- Streamlit-based simple and interactive UI.  
- Latency optimization: context truncation and controlled chunk retrieval.  

---

## Technologies Used

- **Qwen 1.8B** – Language model for response generation.  
- **Ollama Embeddings (nomic-embed-text:v1.5)** – Convert text into vector representations.  
- **FAISS** – Vector database for efficient similarity search.  
- **pdfplumber** – Improved PDF parsing.  
- **docx / txt parsing** – Extract text from Word and text files.  
- **LangChain** – Orchestrates model, embeddings, and vector store.  
- **Streamlit** – Lightweight web interface.  

## Installation

### 1. Clone the Repository

```
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install Python Requirements

```
pip install -r requirements.txt
```

### 3. Install Ollama

Download and install Ollama from:
https://ollama.com

### 4. Pull Required Models

After installing Ollama, run the following commands:

```
ollama pull qwen:1.8b
ollama pull nomic-embed-text:v1.5
```

## Running the Application

Start the Streamlit application:

```
streamlit run app.py
```

## How It Works

1. The user uploads documents.
2. The application extracts text from the documents.
3. The text is split into smaller chunks.
4. Each chunk is converted into embeddings using Granite Embeddings.
5. Embeddings are stored in a FAISS vector store.
6. When a user asks a question, the system retrieves the most relevant chunks.
7. TinyLlama uses this retrieved context to generate an answer.

## Requirements

* Python 3.9+
* Ollama installed
* Required Python libraries listed in `requirements.txt`

## Notes

* The application currently supports document-based question answering.
* Models must be pulled locally using Ollama before running the app.

