# Testing and Validation Guide

## ✅ System Validation Checklist

Use this guide to verify that your Agentic RAG System is working correctly.

---

## 🔍 Pre-Setup Validation

### 1. Environment Check
```bash
# Verify Python version
python --version
# Expected: Python 3.8+

# Verify project structure
cd "f:\Selise Assessment"
ls  # or dir on Windows
# Expected to see: src/, documents/, app.py, cli.py, etc.
```

### 2. Dependencies Check
```bash
pip list | findstr "google-generativeai streamlit faiss numpy"
# Expected: All packages installed

# If missing, install:
pip install -r requirements.txt
```

### 3. API Key Check
```bash
# Verify .env file exists
type .env
# Should contain: GEMINI_API_KEY=your-key-here

# Or set environment variable
$env:GEMINI_API_KEY = "your-key"
```

---

## 🧪 Component Testing

### Test 1: Document Processor
```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
chunks = processor.process_documents("documents")

# Validate
assert len(chunks) > 0, "No chunks created"
assert all("content" in c for c in chunks), "Missing content"
assert all("metadata" in c for c in chunks), "Missing metadata"

print(f"✓ Document Processing: {len(chunks)} chunks created")
```

**Expected Output:**
```
Document Processing: 2 chunks created
✓ Document processor working correctly
```

### Test 2: Embedding Manager
```python
from src.embeddings import EmbeddingManager
import os

api_key = os.getenv("GEMINI_API_KEY")
manager = EmbeddingManager(api_key=api_key)

# Single embedding
embedding = manager.embed_text("What is machine learning?")

# Validate
assert len(embedding) == 768, "Wrong embedding dimension"
assert all(isinstance(x, (int, float)) for x in embedding), "Invalid types"

print(f"✓ Embedding Manager: Generated 768-dim embedding")

# Batch embedding
texts = ["Test 1", "Test 2", "Test 3"]
embeddings = manager.embed_batch(texts)

assert len(embeddings) == 3, "Wrong number of embeddings"
print(f"✓ Batch Embedding: {len(embeddings)} embeddings generated")
```

**Expected Output:**
```
✓ Embedding Manager: Generated 768-dim embedding
✓ Batch Embedding: 3 embeddings generated
```

### Test 3: FAISS Vector Store
```python
from src.embeddings import FAISSVectorStore
from src.document_processor import DocumentProcessor
import os

# Process documents
processor = DocumentProcessor()
chunks = processor.process_documents("documents")

# Create vector store
vector_store = FAISSVectorStore()

# Create dummy embeddings for testing
import numpy as np
dummy_embeddings = [np.random.randn(768).tolist() for _ in chunks]

# Add documents
vector_store.add_documents(chunks, dummy_embeddings)

# Test search
query_embedding = np.random.randn(768).tolist()
results = vector_store.search(query_embedding, top_k=3)

# Validate
assert len(results) <= 3, "Too many results"
assert all(len(r) == 2 for r in results), "Wrong result format"

print(f"✓ FAISS Vector Store: {len(results)} results retrieved")

# Test persistence
vector_store.save()
assert os.path.exists("vector_store/faiss.index"), "Index not saved"
assert os.path.exists("vector_store/metadata.pkl"), "Metadata not saved"

print("✓ Vector Store Persistence: Data saved successfully")
```

**Expected Output:**
```
✓ FAISS Vector Store: 3 results retrieved
✓ Vector Store Persistence: Data saved successfully
```

### Test 4: Retriever
```python
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager, FAISSVectorStore
from src.retriever import RAGRetriever
import os

# Initialize components
api_key = os.getenv("GEMINI_API_KEY")
processor = DocumentProcessor()
chunks = processor.process_documents("documents")

embedding_manager = EmbeddingManager(api_key=api_key)
texts = [c["content"] for c in chunks]
embeddings = embedding_manager.embed_batch(texts)

vector_store = FAISSVectorStore()
vector_store.add_documents(chunks, embeddings)

# Create retriever
retriever = RAGRetriever(
    embedding_manager=embedding_manager,
    vector_store=vector_store,
    top_k=3
)

# Test retrieval
query = "What is machine learning?"
results = retriever.retrieve(query)

# Validate
assert len(results) > 0, "No results retrieved"
assert all(len(r) == 3 for r in results), "Wrong result structure"

print(f"✓ Retriever: {len(results)} documents retrieved")
for i, (content, sim, source) in enumerate(results, 1):
    print(f"  {i}. {source} (similarity: {sim:.2%})")
```

**Expected Output:**
```
✓ Retriever: 2 documents retrieved
  1. ml_fundamentals.txt (similarity: 65.23%)
  2. rag_guide.txt (similarity: 42.15%)
```

### Test 5: Agentic RAG
```python
from src.agent import AgenticRAG
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager, FAISSVectorStore
from src.retriever import RAGRetriever
import os

# Setup
api_key = os.getenv("GEMINI_API_KEY")
processor = DocumentProcessor()
chunks = processor.process_documents("documents")

embedding_manager = EmbeddingManager(api_key=api_key)
texts = [c["content"] for c in chunks]
embeddings = embedding_manager.embed_batch(texts)

vector_store = FAISSVectorStore()
vector_store.add_documents(chunks, embeddings)

retriever = RAGRetriever(
    embedding_manager=embedding_manager,
    vector_store=vector_store
)

# Create agent
agent = AgenticRAG(api_key=api_key, retriever=retriever)

# Test reasoning
query = "What is the difference between supervised and unsupervised learning?"
result = agent.reason(query)

# Validate
assert "answer" in result, "No answer generated"
assert result["confidence"] >= 0 and result["confidence"] <= 100, "Invalid confidence"
assert result["iterations"] > 0, "No iterations recorded"

print(f"✓ Agentic RAG:")
print(f"  - Answer generated: {len(result['answer'])} characters")
print(f"  - Confidence: {result['confidence']}%")
print(f"  - Iterations: {result['iterations']}")
print(f"\nSample Answer: {result['answer'][:200]}...")
```

**Expected Output:**
```
✓ Agentic RAG:
  - Answer generated: 456 characters
  - Confidence: 82%
  - Iterations: 1

Sample Answer: Supervised learning uses labeled data where...
```

---

## 🌐 Interface Testing

### Test Web UI
```bash
# Start Streamlit
streamlit run app.py

# In browser, test:
# 1. Process Documents tab
#    - Upload a .txt file
#    - Click Process
#    - Should show success message

# 2. Ask Questions tab
#    - Enter query: "What is machine learning?"
#    - Should get a response
#    - Check metadata

# 3. System Info tab
#    - Should show configuration
#    - Should show document count
#    - Should show vector store status
```

### Test CLI
```bash
# Show system info
python cli.py info
# Should show: Documents, Vector store status, Configuration

# Process documents (if not already done)
python cli.py process
# Should show: Progress, success message

# Start chat
python cli.py chat
# Should show: Ready for input
# Try: "Tell me about machine learning"
# Type: "quit" to exit
```

---

## 📊 Performance Testing

### Test Document Processing Speed
```python
import time
from src.document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=1000)

start = time.time()
chunks = processor.process_documents("documents")
duration = time.time() - start

print(f"✓ Document Processing Speed:")
print(f"  - Time: {duration:.2f} seconds")
print(f"  - Chunks: {len(chunks)}")
print(f"  - Rate: {len(chunks)/duration:.1f} chunks/sec")

# Expected: Under 5 seconds for sample documents
assert duration < 5, "Processing too slow"
```

### Test Query Latency
```python
import time
from src.agent import AgenticRAG

# Assuming agent is already initialized
agent = AgenticRAG(...)

queries = [
    "What is machine learning?",
    "What is RAG?",
    "How do embeddings work?"
]

times = []
for query in queries:
    start = time.time()
    result = agent.reason(query)
    duration = time.time() - start
    times.append(duration)
    print(f"✓ Query latency: {duration:.2f}s - {query}")

avg_time = sum(times) / len(times)
print(f"\n✓ Average query time: {avg_time:.2f} seconds")

# Expected: 5-30 seconds per query (API dependent)
```

---

## 🎯 Correctness Testing

### Test Answer Quality
```python
from src.agent import AgenticRAG

agent = AgenticRAG(...)

# Test cases: (query, expected_keywords)
test_cases = [
    (
        "What are types of machine learning?",
        ["supervised", "unsupervised", "reinforcement"]
    ),
    (
        "What is RAG?",
        ["retrieval", "augmented", "generation"]
    ),
    (
        "What are embeddings?",
        ["vectors", "semantic", "similarity"]
    )
]

print("✓ Answer Quality Testing:")
for query, keywords in test_cases:
    result = agent.reason(query)
    answer = result["answer"].lower()
    
    found_keywords = [k for k in keywords if k in answer]
    coverage = len(found_keywords) / len(keywords) * 100
    
    print(f"\nQuery: {query}")
    print(f"  - Expected keywords: {', '.join(keywords)}")
    print(f"  - Found: {', '.join(found_keywords)}")
    print(f"  - Coverage: {coverage:.0f}%")
    print(f"  - Confidence: {result['confidence']}%")

# All should have good coverage and confidence
```

**Expected Output:**
```
✓ Answer Quality Testing:

Query: What are types of machine learning?
  - Expected keywords: supervised, unsupervised, reinforcement
  - Found: supervised, unsupervised, reinforcement
  - Coverage: 100%
  - Confidence: 85%

Query: What is RAG?
  - Expected keywords: retrieval, augmented, generation
  - Found: retrieval, augmented, generation
  - Coverage: 100%
  - Confidence: 88%
```

### Test Multi-turn Conversation
```python
from src.agent import AgenticRAG

agent = AgenticRAG(...)

conversation = [
    "What is machine learning?",
    "Explain supervised learning",
    "Give examples of supervised learning algorithms"
]

print("✓ Multi-turn Conversation Test:")
for i, query in enumerate(conversation, 1):
    response = agent.chat(query)
    print(f"\n{i}. User: {query}")
    print(f"   Agent: {response[:100]}...")

# Verify history
history = agent.get_conversation_history()
assert len(history) == len(conversation) * 2, "History incomplete"
print(f"\n✓ Conversation history: {len(history)} messages")
```

---

## 🔒 Error Handling Testing

### Test Missing Documents
```python
import os
from src.document_processor import DocumentProcessor

processor = DocumentProcessor()
# Try to process non-existent directory
chunks = processor.process_documents("non_existent")

assert chunks == [], "Should return empty list"
print("✓ Handles missing documents gracefully")
```

### Test Invalid API Key
```python
from src.embeddings import EmbeddingManager

try:
    manager = EmbeddingManager(api_key="invalid_key")
    embedding = manager.embed_text("test")
    assert False, "Should have raised error"
except Exception as e:
    print(f"✓ Handles invalid API key: {type(e).__name__}")
```

### Test Empty Query
```python
from src.agent import AgenticRAG

agent = AgenticRAG(...)
result = agent.reason("")

# Should handle gracefully
assert "answer" in result
print("✓ Handles empty queries gracefully")
```

---

## ✅ Complete Validation Suite

Run this comprehensive test:

```python
#!/usr/bin/env python
"""
Comprehensive validation test for Agentic RAG System
"""
import os
import sys
from pathlib import Path

def test_imports():
    """Test all imports work"""
    try:
        from src.config import GEMINI_API_KEY
        from src.document_processor import DocumentProcessor
        from src.embeddings import EmbeddingManager, FAISSVectorStore
        from src.retriever import RAGRetriever
        from src.agent import AgenticRAG
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration"""
    try:
        from src.config import GEMINI_API_KEY, CHUNK_SIZE, DOCUMENTS_PATH
        assert GEMINI_API_KEY, "API key not set"
        assert CHUNK_SIZE > 0, "Invalid chunk size"
        assert Path(DOCUMENTS_PATH).exists(), "Documents folder missing"
        print("✓ Configuration valid")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_documents():
    """Test documents exist"""
    try:
        from src.config import DOCUMENTS_PATH
        docs = list(Path(DOCUMENTS_PATH).glob("*.txt"))
        assert len(docs) > 0, "No documents found"
        print(f"✓ Documents found: {len(docs)}")
        return True
    except Exception as e:
        print(f"✗ Document error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("Agentic RAG System - Validation Tests")
    print("="*50)
    
    tests = [
        test_imports,
        test_config,
        test_documents,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Unexpected error in {test.__name__}: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    print("="*50)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

Running this:
```bash
python validate.py
```

**Expected Output:**
```
==================================================
Agentic RAG System - Validation Tests
==================================================
✓ All imports successful
✓ Configuration valid
✓ Documents found: 2

==================================================
Tests Passed: 3/3
==================================================
```

---

## 📈 Expected Results Summary

| Component | Test | Expected Result |
|-----------|------|-----------------|
| Document Processor | Load 2 files | 20+ chunks |
| Embedding | Single text | 768-dim vector |
| FAISS | Search | 3-5 documents |
| Retriever | Query | Ranked documents |
| Agent | Reason | Answer + metadata |
| Web UI | Browse | 3 tabs load |
| CLI | Process | Success message |
| Latency | Single query | 8-20 seconds |

---

## 🎯 Validation Complete!

If all tests pass, your system is:
- ✅ Correctly installed
- ✅ Properly configured
- ✅ Ready for production use
- ✅ Functioning as expected

---

## 🆘 If Tests Fail

1. **Import errors**: Reinstall dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration errors**: Check `.env` file
   ```
   GEMINI_API_KEY=your-actual-key
   ```

3. **Document errors**: Add files to `documents/` folder

4. **API errors**: Verify API key and rate limits

---

For more help, see:
- README.md
- QUICKSTART.md
- ARCHITECTURE.md
