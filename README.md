# 📚 RAG-based Document Q&A Assistant

A Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain. This application allows users to upload PDF documents, process them into embeddings, and ask context-aware questions using the Google Gemini API.

## 🚀 Features
* **PDF Document Upload:** Upload any PDF document directly through the user interface.
* **Intelligent Chunking:** Automatically splits large documents into manageable text chunks using `RecursiveCharacterTextSplitter`.
* **Vector Database:** Uses **FAISS** (Facebook AI Similarity Search) to store and retrieve document embeddings locally and efficiently.
* **Google Gemini Integration:** Utilizes `models/text-embedding-004` for creating embeddings and `gemini-1.5-flash` for generating highly accurate answers.
* **Interactive UI:** Built with Streamlit for a clean, responsive chat interface.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **Orchestration:** LangChain / LangChain Classic
* **Vector Database:** FAISS
* **LLM & Embeddings:** Google Generative AI (Gemini API)
* **Document Parsing:** PyPDF

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git)
cd YOUR-REPOSITORY-NAME
