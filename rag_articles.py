"""
RAG System for Querying Personal Articles
Uses FAISS + HuggingFace Embeddings + Claude Sonnet 4.5
"""

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re

# Load environment variables from .env file
load_dotenv()


def create_vector_store(pdf_folder_path, save_path="./faiss_articles_db"):
    """
    Create and save a FAISS vector store from PDF articles.
    
    Args:
        pdf_folder_path: Path to folder containing PDF files
        save_path: Path where vector store will be saved
        
    Returns:
        vectorstore: FAISS vector store object
    """
    print("📄 Loading PDFs from folder...")
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    documents = loader.load()
    print(f"   ✓ Loaded {len(documents)} PDF documents")
    
    # Show which PDFs were loaded
    pdf_files = set()
    for doc in documents:
        pdf_files.add(os.path.basename(doc.metadata.get('source', 'Unknown')))
    print(f"   PDFs: {', '.join(sorted(pdf_files))}")
    
    print("\n✂️  Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,           # ~1000 chars = 1-3 paragraphs
        chunk_overlap=200,         # 200 char overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]  # Split on logical boundaries
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   ✓ Created {len(chunks)} text chunks")
    
    print("\n🧮 Creating embeddings (this may take a minute on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("   ✓ Embedding model loaded")
    
    print("\n💾 Building FAISS vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("   ✓ Vector index created")
    
    print(f"\n💿 Saving vector store to {save_path}...")
    vectorstore.save_local(save_path)
    print("   ✓ Vector store saved to disk")
    
    print("\n✅ Vector store created successfully!\n")
    return vectorstore


def load_vector_store(load_path="./faiss_articles_db"):
    """
    Load existing FAISS vector store from disk.
    
    Args:
        load_path: Path to saved vector store
        
    Returns:
        vectorstore: FAISS vector store object
    """
    print("📂 Loading existing vector store...")
    
    # Must use same embedding model as when created
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.load_local(
        load_path,
        embeddings,
        allow_dangerous_deserialization=True  # Required for FAISS
    )
    
    print("   ✓ Vector store loaded successfully!\n")
    return vectorstore


def create_qa_chain(vectorstore, retrieval_k=4):
    """
    Create a RetrievalQA chain with Claude Sonnet 4.5.
    
    Args:
        vectorstore: FAISS vector store for retrieval
        retrieval_k: Number of chunks to retrieve (default 4)
        
    Returns:
        qa_chain: RetrievalQA chain ready for queries
    """
    print("🤖 Setting up Claude Sonnet 4.5...")
    
    # Initialize Claude
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0  # Deterministic responses
    )
    
    # Custom prompt template
    prompt_template = """You are helping the user explore their own written articles.
Use the following context from their articles to answer the question thoughtfully.

Guidelines:
- If you reference specific information, mention which article it comes from
- If you're not certain about something, say so honestly
- Provide detailed, helpful answers based on the context
- If the context doesn't contain relevant information, say so
- When asked to list articles or titles, extract ALL article titles you can find in the context

Context from articles:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Stuff all retrieved docs into context
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_k}  # Retrieve top k most relevant chunks
        ),
        return_source_documents=True,  # Return source docs for transparency
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print(f"   ✓ QA chain configured (retrieving top {retrieval_k} chunks)\n")
    return qa_chain


def extract_all_titles(vectorstore):
    """
    Extract all unique article titles from the vector store.
    Uses heuristics to identify titles from the document chunks.
    
    Args:
        vectorstore: FAISS vector store
        
    Returns:
        list: List of article titles found
    """
    # Get all documents from vector store
    all_docs = vectorstore.docstore._dict.values()
    
    titles = set()
    
    # Common title patterns in academic papers
    title_patterns = [
        r'^[A-Z][^.!?]*(?:in|of|for|with|at|on|and|the)\s+[^.!?]*$',  # Title case with prepositions
        r'^\*\*.*\*\*$',  # Markdown bold
        r'^# .*$',  # Markdown header
    ]
    
    for doc in all_docs:
        content = doc.page_content
        lines = content.split('\n')
        
        # Check first few lines for titles
        for line in lines[:5]:
            line = line.strip()
            
            # Skip empty lines or very short lines
            if len(line) < 10 or len(line) > 200:
                continue
            
            # Check if line looks like a title
            # Heuristics: starts with capital, no ending punctuation, reasonable length
            if (line[0].isupper() and 
                not line.endswith('.') and 
                not line.endswith('?') and
                not line.endswith('!') and
                20 < len(line) < 200):
                titles.add(line)
    
    return sorted(list(titles))


def get_all_source_files(vectorstore):
    """
    Get all unique PDF source files from the vector store.
    
    Args:
        vectorstore: FAISS vector store
        
    Returns:
        list: List of PDF filenames
    """
    all_docs = vectorstore.docstore._dict.values()
    sources = set()
    
    for doc in all_docs:
        source = doc.metadata.get('source', '')
        if source:
            sources.add(os.path.basename(source))
    
    return sorted(list(sources))


def query_articles(qa_chain, question):
    """
    Query the RAG system with a single question.
    Automatically detects comprehensive queries and retrieves more chunks.
    
    Args:
        qa_chain: RetrievalQA chain
        question: Question string
        
    Returns:
        dict: Result with 'answer' and 'sources'
    """
    # Detect if user wants a comprehensive list
    list_keywords = ['list all', 'all titles', 'all articles', 'all my', 'every article', 
                     'complete list', 'show all', 'what articles', 'which articles']
    wants_comprehensive_list = any(keyword in question.lower() for keyword in list_keywords)
    
    if wants_comprehensive_list:
        # Temporarily modify the retriever to get more chunks
        original_k = qa_chain.retriever.search_kwargs.get('k', 4)
        qa_chain.retriever.search_kwargs['k'] = 20  # Retrieve 20 chunks for comprehensive queries
        result = qa_chain({"query": question})
        qa_chain.retriever.search_kwargs['k'] = original_k  # Restore original
    else:
        result = qa_chain({"query": question})
    
    # Extract unique source files
    sources = set()
    if result.get('source_documents'):
        for doc in result['source_documents']:
            source = doc.metadata.get('source', 'Unknown')
            sources.add(os.path.basename(source))
    
    return {
        'answer': result['result'],
        'sources': list(sources)
    }


def query_with_all_context(vectorstore, question):
    """
    Query using ALL available context from vector store (no retrieval limit).
    Warning: This can be slow and expensive for large document sets.
    
    Args:
        vectorstore: FAISS vector store
        question: Question string
        
    Returns:
        dict: Result with 'answer' and 'sources'
    """
    print("   (Using ALL documents - this may take a moment...)")
    
    # Get all documents
    all_docs = list(vectorstore.docstore._dict.values())
    
    # Combine all content
    all_text = "\n\n---\n\n".join([doc.page_content for doc in all_docs[:50]])  # Limit to first 50 to avoid token limits
    
    # Initialize Claude
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0
    )
    
    prompt = f"""Based on ALL of the following context from the user's articles, answer this question:

Question: {question}

Context:
{all_text}

Answer:"""
    
    answer = llm.invoke(prompt).content
    
    # Get all sources
    sources = get_all_source_files(vectorstore)
    
    return {
        'answer': answer,
        'sources': sources,
        'num_chunks_retrieved': len(all_docs)
    }


def chat_with_articles(qa_chain, vectorstore):
    """
    Interactive chat loop for querying articles.
    
    Args:
        qa_chain: RetrievalQA chain
        vectorstore: FAISS vector store for enhanced queries
    """
    print("="*70)
    print("💬 Chat with Your Articles")
    print("="*70)
    print("Ask questions about your articles, or type 'quit' to exit")
    print("\nSpecial commands:")
    print("  'list files' - Show all PDF files in the database")
    print("  'stats' - Show database statistics")
    print("\nExample questions:")
    print("  - What did I write about machine learning?")
    print("  - List all titles of my articles")
    print("  - Summarize my thoughts on startups")
    print("  - What tools or frameworks did I mention?")
    print("="*70 + "\n")
    
    while True:
        question = input("You: ").strip()
        
        # Exit commands
        if question.lower() in ['quit', 'exit', 'q', 'bye']:
            print("\n👋 Goodbye!")
            break
        
        # Special commands
        if question.lower() == 'list files':
            files = get_all_source_files(vectorstore)
            print(f"\n📚 Found {len(files)} PDF files:")
            for i, f in enumerate(files, 1):
                print(f"   {i}. {f}")
            print()
            continue
        
        if question.lower() == 'stats':
            files = get_all_source_files(vectorstore)
            all_docs = list(vectorstore.docstore._dict.values())
            print(f"\n📊 Database Statistics:")
            print(f"   PDF files: {len(files)}")
            print(f"   Total chunks: {len(all_docs)}")
            print(f"   Avg chunk size: {sum(len(d.page_content) for d in all_docs) // len(all_docs)} chars")
            print()
            continue
            
        if not question:
            continue
        
        print("\n🤔 Thinking...\n")
        
        try:
            # Query the system
            result = query_articles(qa_chain, question, vectorstore)
            
            # Display answer
            print(f"Claude: {result['answer']}")
            
            # Display metadata
            print(f"\n📊 Retrieved {result['num_chunks_retrieved']} chunks from {len(result['sources'])} files")
            if result['sources']:
                print(f"📚 Sources: {', '.join(result['sources'][:5])}")
                if len(result['sources']) > 5:
                    print(f"   ... and {len(result['sources']) - 5} more")
            
            print("\n" + "-"*70 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try rephrasing your question.\n")

