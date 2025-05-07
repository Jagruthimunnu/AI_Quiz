"""
vector_db_setup.py - Utility to initialize the vector database for ollamachat.py

This script extracts text from PDFs, creates document chunks, and builds a 
Chroma vector database using the same models and structure as ollamachat.py.
"""

import os
import sys
import fitz  # PyMuPDF for PDF extraction
import pandas as pd
import csv
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def setup_directories(base_dir):
    """Create necessary directories if they don't exist."""
    try:
        # Create base directory
        os.makedirs(base_dir, exist_ok=True)
        
        # Create directory for PDF storage
        pdf_dir = os.path.join(base_dir, "maths")
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Create directory for vector store
        db_dir = os.path.join(base_dir, "chroma_langchain_db")
        os.makedirs(db_dir, exist_ok=True)
        
        print(f"✓ All directories created/verified at {base_dir}")
        return True
    except Exception as e:
        print(f"✗ Error creating directories: {str(e)}")
        return False

def extract_text_from_pdf(pdf_path, csv_path):
    """Extract text from a PDF file, split into chunks, and save to CSV."""
    if not os.path.exists(pdf_path):
        print(f"✗ Error: PDF file not found at {pdf_path}")
        return None
    
    chunks = []
    try:
        print(f"Processing PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Write to CSV file
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['source', 'page', 'chunk', 'content']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()
                # Split text into smaller paragraphs
                paragraphs = [p for p in text.split('\n\n') if p.strip()]
                
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        # Create a chunk dictionary
                        chunk = {
                            "content": paragraph,
                            "metadata": {
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1,
                                "chunk": i + 1
                            }
                        }
                        chunks.append(chunk)
                        
                        # Write to CSV
                        writer.writerow({
                            'source': os.path.basename(pdf_path),
                            'page': page_num + 1,
                            'chunk': i + 1,
                            'content': paragraph
                        })
        
        print(f"✓ Successfully extracted text from PDF and saved to {csv_path}")
        print(f"  Total chunks: {len(chunks)}")
        
        return chunks
    except Exception as e:
        print(f"✗ Error extracting text from PDF: {str(e)}")
        return None

def create_vector_store(chunks, db_location):
    """Create and populate the vector store with document chunks."""
    try:
        print("Initializing embedding model (mxbai-embed-large)...")
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        
        # Create directory for vector store if it doesn't exist
        os.makedirs(db_location, exist_ok=True)
        
        print(f"Creating vector store at {db_location}...")
        vector_store = Chroma(
            collection_name="pdf_content",
            persist_directory=db_location,
            embedding_function=embeddings
        )
        
        if chunks:
            documents = []
            ids = []
            
            print(f"Preparing {len(chunks)} documents for vector database...")
            for i, chunk in enumerate(chunks):
                document = Document(
                    page_content=chunk["content"],
                    metadata=chunk["metadata"],
                )
                ids.append(str(i))
                documents.append(document)
            
            # Add documents in batches to avoid memory issues
            batch_size = 50
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(documents))
                
                batch_docs = documents[start_idx:end_idx]
                batch_ids = ids[start_idx:end_idx]
                
                print(f"Adding batch {batch_num + 1}/{total_batches} ({len(batch_docs)} documents)...")
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            
            print("Persisting vector store...")
            vector_store.persist()
            print(f"✓ Vector store created successfully with {len(documents)} documents")
            return True
        else:
            print("✗ No documents to add to vector store")
            return False
            
    except Exception as e:
        print(f"✗ Error creating vector store: {str(e)}")
        return False

def test_vector_store(db_location, query="basic mathematics concepts"):
    """Test the vector store with a simple query."""
    try:
        print("\nTesting vector store...")
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        
        vector_store = Chroma(
            collection_name="pdf_content",
            persist_directory=db_location,
            embedding_function=embeddings
        )
        
        print(f"Running test query: '{query}'")
        results = vector_store.similarity_search(query, k=2)
        
        print(f"✓ Retrieved {len(results)} results")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1} (Page {doc.metadata['page']}, Chunk {doc.metadata['chunk']}):")
            print(f"Content: {doc.page_content[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ Error testing vector store: {str(e)}")
        return False

def main():
    """Main function to set up the vector database."""
    # Set base directory - use the same paths as ollamachat.py
    BASE_DIR = "/Users/jagruthimekala/Library/Mobile Documents/com~apple~CloudDocs/rag"
    if len(sys.argv) > 1:
        BASE_DIR = sys.argv[1]
    
    print(f"Setting up vector database for ollamachat.py")
    print(f"Base directory: {BASE_DIR}")
    
    # Create directories
    if not setup_directories(BASE_DIR):
        return
    
    # Define paths
    PDF_DIR = os.path.join(BASE_DIR, "10th Maths - NCERT")

    PDF_PATH = os.path.join(PDF_DIR, "maths2.pdf")
    DB_LOCATION = os.path.join(BASE_DIR, "chroma_langchain_db")
    CSV_PATH = os.path.join(BASE_DIR, "extracted_text.csv")
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"✗ Warning: PDF file not found at {PDF_PATH}")
        print(f"  Please place your PDF file at this location before proceeding.")
        return
    
    # Extract text from PDF
    chunks = extract_text_from_pdf(PDF_PATH, CSV_PATH)
    if not chunks:
        return
    
    # Create vector store
    if not create_vector_store(chunks, DB_LOCATION):
        return
    
    # Test vector store
    test_vector_store(DB_LOCATION)
    
    print("\n✓ Vector database setup complete!")
    print(f"  - PDF processed: {PDF_PATH}")
    print(f"  - CSV created: {CSV_PATH}")
    print(f"  - Vector DB: {DB_LOCATION}")
    print("\nYou can now run ollamachat.py to use the quiz application.")

if __name__ == "__main__":
    main()