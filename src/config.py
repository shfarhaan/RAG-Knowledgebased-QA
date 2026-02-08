"""
Configuration settings for the Agentic RAG System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "DNXsUJ6yvjFrLFXoptvfWK4qZfmeUYlFxl4HwIqr6oaQAuBPhYoQJQQJ99CBACHYHv6XJ3w3AAAAACOGHYOLz")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://assessment-6-temp-resource.cognitiveservices.azure.com")

# Model Configuration
CHAT_DEPLOYMENT_NAME = "gpt-4o-mini"
EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"
CHAT_API_VERSION = "2025-01-01-preview"
EMBEDDING_API_VERSION = "2023-05-15"
TEMPERATURE = 0.3

# Azure OpenAI Full URLs (for convenience)
AZURE_CHAT_URL = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{CHAT_DEPLOYMENT_NAME}/chat/completions?api-version={CHAT_API_VERSION}"
AZURE_EMBEDDING_URL = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{EMBEDDING_DEPLOYMENT_NAME}/embeddings?api-version={EMBEDDING_API_VERSION}"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Store Configuration
VECTOR_STORE_PATH = "vector_store/faiss_index"
EMBEDDINGS_MODEL = "azure-openai"  # Using Azure OpenAI embeddings

# Document Configuration
DOCUMENTS_PATH = "documents"
SUPPORTED_FORMATS = [".txt", ".pdf", ".md"]

# Agent Configuration
MAX_RETRIEVAL_ATTEMPTS = 3
TOP_K_DOCUMENTS = 5
SIMILARITY_THRESHOLD = 0.3

# Streamlit Configuration
STREAMLIT_PAGE_TITLE = "Agentic RAG System for Domain Knowledge QA"
STREAMLIT_PAGE_ICON = "🤖"
