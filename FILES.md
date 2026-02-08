# Files and Components Reference

## 📁 Project Structure

```
f:\Selise Assessment/
│
├── 📄 Core Application Files
│   ├── app.py                    # Main Streamlit web application
│   ├── cli.py                    # Command-line interface
│   └── notebook_example.ipynb    # Jupyter notebook with examples
│
├── 📚 Source Code (src/)
│   ├── __init__.py              # Python package initialization
│   ├── config.py                # Configuration and constants
│   ├── document_processor.py    # Document loading & chunking
│   ├── embeddings.py            # Embedding generation & FAISS
│   ├── retriever.py             # Retrieval logic
│   └── agent.py                 # Agentic RAG with reflection
│
├── 📁 Documents
│   ├── ml_fundamentals.txt      # Sample: ML concepts
│   └── rag_guide.txt            # Sample: RAG systems guide
│
├── 📁 Vector Store (auto-created)
│   ├── faiss.index              # FAISS vector index
│   └── metadata.pkl             # Document metadata
│
├── 📖 Documentation
│   ├── README.md                # Complete system documentation
│   ├── QUICKSTART.md            # Quick start guide
│   ├── ARCHITECTURE.md          # System architecture & design
│   └── FILES.md                 # This file
│
├── ⚙️ Configuration
│   ├── requirements.txt         # Python dependencies
│   ├── .env.example             # API key template
│   └── .env                     # Your actual API keys (IGNORED)
│
└── 🚀 Runtime Directories (auto-created)
    └── vector_store/            # FAISS indices and metadata
```

## 📋 File Descriptions

### Application Files

#### `app.py` (341 lines)
**Streamlit web application**

Components:
- `initialize_rag_system()`: Cached RAG initialization
- `process_documents_tab()`: Document upload & processing UI
- `qa_chat_tab()`: Q&A chat interface
- `system_info_tab()`: System status & information
- `main()`: Application entry point

Features:
- Document processing with progress indicators
- Interactive chat with conversation history
- Metadata display (confidence, iterations)
- System status checks
- Custom CSS styling

Usage:
```bash
streamlit run app.py
```

#### `cli.py` (358 lines)
**Command-line interface**

Commands:
- `process`: Load and chunk documents, generate embeddings
- `chat`: Start interactive terminal chat session
- `info`: Show system configuration and status
- `help`: Display help information

Features:
- Colored terminal output
- Interactive chat loop with history
- Document processing progress tracking
- System information display

Usage:
```bash
python cli.py process  # Process documents
python cli.py chat     # Start chatting
python cli.py info     # Show info
```

#### `notebook_example.ipynb` (8 cells)
**Jupyter notebook demonstration**

Contains:
1. Setup and imports
2. Configuration loading
3. Document processing demo
4. Embedding generation
5. Vector store creation
6. Retriever testing
7. Agentic RAG examples
8. Multi-turn conversation
9. History analysis

Use for:
- Learning system components
- Step-by-step execution
- Interactive exploration
- Debugging

Usage:
```bash
jupyter notebook notebook_example.ipynb
```

---

### Source Code (src/)

#### `__init__.py` (18 lines)
**Package initialization**

Exports:
- `DocumentProcessor`
- `EmbeddingManager`
- `FAISSVectorStore`
- `RAGRetriever`
- `AgenticRAG`

#### `config.py` (34 lines)
**Configuration management**

Constants:
- API Keys: `GEMINI_API_KEY`
- Models: `EMBEDDING_MODEL`, `LLM_MODEL`
- Parameters: `TEMPERATURE`, `CHUNK_SIZE`, etc.
- Paths: `DOCUMENTS_PATH`, `VECTOR_STORE_PATH`
- Thresholds: `SIMILARITY_THRESHOLD`, `MAX_RETRIEVAL_ATTEMPTS`

Modify for:
- API credentials
- Model selection
- Chunk sizes
- Retrieval settings

#### `document_processor.py` (103 lines)
**Document processing pipeline**

Classes:
- `DocumentProcessor`: Main processor class

Methods:
- `load_documents(directory)`: Load files
- `chunk_text(text)`: Split text into chunks
- `process_documents(directory)`: Full pipeline

Features:
- Multi-format support (.txt, .md, .pdf)
- Intelligent sentence-aware chunking
- Metadata preservation
- Error handling
- Logging

Output:
```python
[
    {
        "content": "...",
        "source": "filename.txt",
        "chunk_id": 0,
        "metadata": {...}
    }
]
```

#### `embeddings.py` (169 lines)
**Embedding generation and vector storage**

Classes:
- `EmbeddingManager`: Google API client for embeddings
- `FAISSVectorStore`: Local FAISS vector database

EmbeddingManager Methods:
- `embed_text(text)`: Single text embedding
- `embed_batch(texts, batch_size)`: Batch processing

FAISSVectorStore Methods:
- `add_documents(chunks, embeddings)`: Add to index
- `search(query_embedding, top_k)`: Similarity search
- `save()`: Persist to disk
- `load()`: Load from disk

Features:
- Google Generative AI API integration
- Batch processing with progress
- FAISS L2 distance indexing
- Persistent storage with pickle
- Metadata association

#### `retriever.py` (80 lines)
**Retrieval logic**

Class:
- `RAGRetriever`: Main retriever

Methods:
- `retrieve(query)`: Get relevant documents
- `retrieve_with_context(query)`: Formatted context string

Features:
- Query embedding generation
- FAISS vector search
- Similarity threshold filtering
- Source attribution
- Context formatting

Output:
```python
[
    (content, similarity_score, source_document),
    ...
]
```

#### `agent.py` (278 lines)
**Agentic RAG with reflection and reasoning**

Classes:
- `AgentState`: Enum for agent states
- `AgenticRAG`: Main agentic RAG system

Methods:
- `reason(query)`: Full reasoning loop
- `chat(user_input)`: Chat interface
- `_tool_retrieve_documents(query)`: Retriever tool
- `_critic_evaluate_retrieved_docs(query, context)`: Evaluation
- `_generate_answer(query, context)`: Answer generation

Features:
- Multi-iteration reasoning
- Tool using (retriever as tool)
- Self-reflection (critic)
- Query refinement
- Confidence scoring
- Conversation history

States:
1. INITIAL: Start
2. RETRIEVING: Call retriever tool
3. REFLECTING: Critic evaluation
4. ANALYZING: Process results
5. GENERATING: LLM generation
6. COMPLETE: Done

---

### Documentation Files

#### `README.md` (380+ lines)
**Complete system documentation**

Sections:
- System architecture diagram
- Feature overview
- Installation steps
- Quick start guide
- Usage guide with examples
- Configuration reference
- Troubleshooting
- Performance optimization
- Evaluation metrics
- Further reading
- Contributing guidelines

#### `QUICKSTART.md` (110+ lines)
**5-minute quick start**

Sections:
- API key setup
- Installation
- Configuration
- Interface options
- Usage examples
- Troubleshooting
- Next steps

For: New users, quick setup

#### `ARCHITECTURE.md` (250+ lines)
**Detailed architecture and design**

Sections:
- Overall architecture diagram
- Data flow stages
- Agent state machine
- Component details
- Execution flow example
- Persistence strategy
- Performance characteristics
- Learning components
- Safety measures

For: Understanding system design, debugging

#### `.env.example` (5 lines)
**Environment variable template**

Contains:
```
GEMINI_API_KEY=your-key-here
```

Create `.env` with your actual keys.

---

### Data Files

#### `documents/ml_fundamentals.txt` (~2000 words)
**Sample document about machine learning**

Topics:
- Introduction to ML
- ML algorithms
- Feature engineering
- Model evaluation
- Overfitting/underfitting
- Deep learning
- Applications
- Best practices

Use for: Testing/demo with ML-related queries

#### `documents/rag_guide.txt` (~3000 words)
**Sample document about RAG systems**

Topics:
- What is RAG
- Advantages of RAG
- Retrieval methods
- Embedding models
- Vector databases
- Chunking strategies
- Ranking/re-ranking
- QA architectures
- Evaluation metrics
- Challenges
- Best practices

Use for: Testing/demo with RAG-related queries

---

### Configuration Files

#### `requirements.txt` (5 lines)
**Python package dependencies**

```
google-generativeai>=0.3.0  # Google Gemini API
streamlit>=1.28.0            # Web UI
faiss-cpu>=1.7.0             # Vector search
numpy>=1.24.0                # Numerical computing
python-dotenv>=1.0.0         # Environment variables
```

Install:
```bash
pip install -r requirements.txt
```

---

## 📊 Component Interactions

```
app.py (Streamlit)
    ↓
    ├─ Uses → config.py (settings)
    ├─ Uses → document_processor.py (loading/chunking)
    ├─ Uses → embeddings.py (embeddings + FAISS)
    ├─ Uses → retriever.py (document retrieval)
    └─ Uses → agent.py (agentic reasoning)

cli.py (CLI)
    ↓
    ├─ Uses → config.py (settings)
    ├─ Uses → document_processor.py (loading/chunking)
    ├─ Uses → embeddings.py (embeddings + FAISS)
    ├─ Uses → retriever.py (document retrieval)
    └─ Uses → agent.py (agentic reasoning)

notebook_example.ipynb
    ↓
    └─ Demonstrates all components step-by-step

Core Components:
    document_processor.py
        ↓
        ├─ Input: Raw documents
        └─ Output: Chunks with metadata
    
    embeddings.py
        ├─ EmbeddingManager: Google API client
        └─ FAISSVectorStore: Local vector DB
    
    retriever.py
        ├─ Input: User query
        └─ Output: Retrieved documents
    
    agent.py
        ├─ Tool: retriever
        ├─ Critic: evaluator
        ├─ Generator: LLM
        └─ Output: Grounded answer
```

---

## 🔄 Data Flow

```
User Input (Query)
    ↓
    ├─ CLI, Web UI, or Notebook interface
    ↓
agent.py::reason()
    ├─ Call retriever tool
    │   └─ Query → Embedding → FAISS Search → Documents
    ├─ Critic evaluation
    │   └─ Assess relevance → Refine or proceed
    ├─ Generate answer
    │   └─ Query + Context → LLM → Response
    └─ Return result with metadata
        ├─ Answer text
        ├─ Confidence score
        ├─ Sources used
        └─ Reasoning steps

Back to user
    ├─ Display answer
    ├─ Show sources
    ├─ Display metadata
    └─ Save to history
```

---

## 📝 File Statistics

| File | Lines | Purpose |
|------|-------|---------|
| app.py | 341 | Streamlit web UI |
| cli.py | 358 | Command-line interface |
| config.py | 34 | Configuration |
| document_processor.py | 103 | Document processing |
| embeddings.py | 169 | Embeddings + FAISS |
| retriever.py | 80 | Retrieval logic |
| agent.py | 278 | Agentic RAG |
| **Total Source** | **1,363** | **Core system** |
| README.md | 380 | Main documentation |
| ARCHITECTURE.md | 250 | Design details |
| QUICKSTART.md | 110 | Quick start guide |
| **Total Docs** | **740** | **Documentation** |

---

## 🚀 Getting Started with Files

### First Time Setup
1. Read: `QUICKSTART.md`
2. Set: `API_KEY` in `.env`
3. Run: `app.py` (Streamlit)

### Understanding the System
1. Read: `README.md` (overview)
2. Read: `ARCHITECTURE.md` (design)
3. Run: `notebook_example.ipynb` (step-by-step)

### Customization
1. Edit: `src/config.py` (settings)
2. Add: Documents to `documents/` folder
3. Run: `app.py` → Process → Chat

### Debugging
1. Check: Logs in terminal
2. Read: Troubleshooting in `README.md`
3. Run: `cli.py info` (status check)

---

## 📚 Key Concepts by File

| Concept | File |
|---------|------|
| Chunking Strategy | document_processor.py |
| Embedding Model | embeddings.py (EmbeddingManager) |
| Vector Database | embeddings.py (FAISSVectorStore) |
| Similarity Search | retriever.py |
| Tool Calling | agent.py (_tool_retrieve_documents) |
| Self-Reflection | agent.py (_critic_evaluate_retrieved_docs) |
| Answer Generation | agent.py (_generate_answer) |
| Chat Interface | app.py (qa_chat_tab) |
| Configuration | config.py |

---

Last Updated: February 2026
