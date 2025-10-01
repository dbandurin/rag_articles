# rag_articles

markdown# RAG Articles - Query Your Personal Research Papers

A Retrieval-Augmented Generation (RAG) system to chat with your collection of PDF articles using AI. Built with LangChain, FAISS, and Claude Sonnet 4.5.

## Features

- 📄 **PDF Processing** - Automatically extracts and indexes content from PDF articles
- 🔍 **Semantic Search** - Uses embeddings to find relevant content based on meaning, not just keywords
- 💬 **Natural Conversations** - Ask questions in plain English about your research
- 🚀 **Fast Retrieval** - FAISS vector database for efficient similarity search
- 🤖 **Powered by Claude** - Uses Claude Sonnet 4.5 for intelligent, context-aware answers
- 🔒 **Local Embeddings** - Embeddings run locally for privacy (only LLM queries use API)

## Installation

### Prerequisites
- Python 3.10+
- Anthropic API key ([get one here](https://console.anthropic.com))

### Setup

1. Clone the repository
```
git clone https://github.com/yourusername/rag-articles.git
cd rag-articles
```
2. Create virtual environment
```
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Configure API key

5. Create a .env file in the project root:
```
ANTHROPIC_API_KEY=your-api-key-here
```

6. Add your PDFs
```
mkdir articles
```

# Copy your PDF files into the articles/ folder
Usage
Command Line Interface
Run the interactive chat:
```
python rag_articles.py
```
On first run, it will process your PDFs and create a vector database. Subsequent runs load the existing database instantly.

Python API: see QA_example.ipynb
