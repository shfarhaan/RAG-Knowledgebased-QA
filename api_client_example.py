#!/usr/bin/env python
"""
Example API Client for Agentic RAG System
Demonstrates how to interact with the REST API
"""
import requests
import json
from typing import Optional

# Base URL for the API
API_URL = "http://localhost:8000"


def check_health() -> dict:
    """Check if API is running"""
    response = requests.get(f"{API_URL}/health")
    return response.json()


def get_info() -> dict:
    """Get system information"""
    response = requests.get(f"{API_URL}/info")
    return response.json()


def upload_document(file_path: str) -> dict:
    """
    Upload a document to the system
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Upload response
    """
    with open(file_path, 'rb') as f:
        files = {'files': f}
        response = requests.post(f"{API_URL}/upload", files=files)
    return response.json()


def process_documents(chunk_size: int = 1000, chunk_overlap: int = 200) -> dict:
    """
    Process uploaded documents
    
    Args:
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Processing response
    """
    data = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }
    response = requests.post(f"{API_URL}/process", json=data)
    return response.json()


def ask_question(question: str, max_iterations: Optional[int] = 3) -> dict:
    """
    Ask a question to the RAG system
    
    Args:
        question: Your question
        max_iterations: Maximum reasoning iterations
        
    Returns:
        Answer with metadata
    """
    data = {
        "question": question,
        "max_iterations": max_iterations
    }
    response = requests.post(f"{API_URL}/chat", json=data)
    return response.json()


def clear_history() -> dict:
    """Clear conversation history"""
    response = requests.post(f"{API_URL}/clear")
    return response.json()


def cleanup_documents() -> dict:
    """Remove uploaded documents from disk"""
    response = requests.delete(f"{API_URL}/cleanup")
    return response.json()


def main():
    """Example usage"""
    
    print("=" * 60)
    print("Agentic RAG API Client Example")
    print("=" * 60)
    
    # 1. Check health
    print("\n1. Checking API health...")
    health = check_health()
    print(f"   Status: {health['status']}")
    
    # 2. Get system info
    print("\n2. Getting system information...")
    info = get_info()
    print(f"   Status: {info['status']}")
    print(f"   Chat Model: {info['chat_model']}")
    print(f"   Vector Store Ready: {info['vector_store_ready']}")
    print(f"   Documents: {info['documents_count']}")
    
    # 3. Ask a question (if vector store is ready)
    if info['vector_store_ready']:
        print("\n3. Asking a question...")
        question = "What is machine learning?"
        
        result = ask_question(question)
        
        print(f"\n   Question: {question}")
        print(f"\n   Answer: {result['answer'][:200]}...")
        print(f"\n   Confidence: {result['confidence']}%")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Sources: {', '.join(result['sources'])}")
    else:
        print("\n3. Vector store not ready. Upload and process documents first.")
        print("   Example:")
        print("   - upload_document('path/to/document.txt')")
        print("   - process_documents()")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server.")
        print("Make sure the API server is running:")
        print("  python api.py")
    except Exception as e:
        print(f"Error: {str(e)}")
