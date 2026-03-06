# Mini RAG Bot

## Overview

This project is a **Mini Retrieval-Augmented Generation (RAG) Bot** that allows users to upload their documents and ask questions related to the uploaded content.

The application processes the documents, converts the text into embeddings, stores them in a vector database, and retrieves the most relevant context when a user asks a question. The retrieved information is then used by a language model to generate a precise answer.

## Features

* Upload documents and extract their text
* Automatically split documents into smaller chunks
* Generate embeddings for each chunk
* Store embeddings in a FAISS vector database
* Retrieve relevant document sections for a query
* Generate answers using a language model
* Simple user interface built with Streamlit

## Technologies Used

* **TinyLlama** – Language model used for generating responses
* **Granite Embeddings** – Used to convert text into vector representations
* **FAISS** – Vector database for similarity search
* **Streamlit** – Lightweight UI for interacting with the application
* **LangChain** – Framework used to integrate models, embeddings, and vector stores

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
ollama pull tinyllama
ollama pull granite-embedding
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

