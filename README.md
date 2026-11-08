# Retrival-augumented-generation-code-
 Project Overview

Retrieval-Augmented Generation (RAG) is an AI architecture that combines information retrieval with Large Language Models (LLMs).

This project implements a PDF-based RAG pipeline where users can:

Upload PDF documents.

Extract text from PDF files.

Split documents into smaller chunks.

Generate embeddings using Hugging Face.

Store embeddings in a FAISS vector database.

Retrieve relevant document chunks using similarity search.

Generate answers using an LLM.

Interact with the system through a Streamlit web interface.

🛠️ Technologies Used

Technology

Purpose

Python

Core programming language

LangChain

Building the RAG pipeline

Hugging Face

Text embedding generation

FAISS

Vector database and similarity search

OpenAI API

LLM-based response generation

PyPDF

PDF text extraction

Streamlit

Interactive web application

python-dotenv

Environment variable management

