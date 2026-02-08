# System Architecture and Design

## 🏗️ Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AGENTIC RAG SYSTEM                          │
└─────────────────────────────────────────────────────────────────────┘

                            USER INTERFACE LAYER
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                ┌───▼────┐  ┌──────▼──────┐  ┌───▼────┐
                │Streamlit│  │  CLI Tool   │  │Notebook│
                │   UI    │  │  Interface  │  │ Demo   │
                └────┬────┘  └──────┬──────┘  └───┬────┘
                     │              │             │
                     └──────────────┼─────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │    APPLICATION LAYER            │
                    │  ┌─────────────────────────┐   │
                    │  │  Document Processor     │   │
                    │  │  - Load documents       │   │
                    │  │  - Chunking strategy    │   │
                    │  │  - Metadata extraction  │   │
                    │  └─────────────────────────┘   │
                    │  ┌─────────────────────────┐   │
                    │  │ Embedding Manager       │   │
                    │  │  - Azure OpenAI client  │   │
                    │  │  - Batch processing     │   │
                    │  │  - Caching              │   │
                    │  └─────────────────────────┘   │
                    │  ┌─────────────────────────┐   │
                    │  │ RAG Retriever           │   │
                    │  │  - Query embedding      │   │
                    │  │  - Similarity search    │   │
                    │  │  - Ranking & filtering  │   │
                    │  └─────────────────────────┘   │
                    │  ┌─────────────────────────┐   │
                    │  │ Agentic RAG             │   │
                    │  │  - Tool calling         │   │
                    │  │  - Self-reflection      │   │
                    │  │  - Answer generation    │   │
                    │  │  - Multi-iteration      │   │
                    │  └─────────────────────────┘   │
                    └───────────────┬─────────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
        ┌───────▼────────┐  ┌───────▼─────┐  ┌──────▼──────┐
        │ External APIs  │  │  Vector DB  │  │  File       │
        │                │  │             │  │  Storage    │
        │ Azure OpenAI   │  │  FAISS      │  │             │
        │ Service        │  │  Index      │  │  Documents  │
        │ - Embeddings   │  │             │  │  Vector     │
        │ - LLM          │  │             │  │  Store      │
        └────────────────┘  └─────────────┘  └─────────────┘

```

## 🔄 Data Flow: From Document to Answer

```
STAGE 1: SETUP & INITIALIZATION
═════════════════════════════════════════════════════════════════

Raw Documents
    │
    ▼
Document Processor
├─ Load files (.txt, .md, .pdf)
├─ Split into chunks (with overlap)
└─ Extract metadata

Chunks
    │
    ▼
Embedding Manager
└─ Convert chunks to vectors (Azure OpenAI API)

Embeddings
    │
    ▼
FAISS Vector Store
├─ Build L2 distance index
├─ Save to disk
└─ Metadata stored


STAGE 2: QUERY TIME - AGENTIC REASONING
═════════════════════════════════════════════════════════════════

User Query: "What is machine learning?"
    │
    ▼
ITERATION LOOP (max 3 iterations):
    │
    ├─ STEP 1: Tool Use - RETRIEVER
    │   ├─ Generate query embedding
    │   ├─ Search FAISS index
    │   ├─ Retrieve top-K documents
    │   └─ Return context
    │
    ├─ STEP 2: Self-Reflection - CRITIC
    │   ├─ Evaluate relevance (0-100)
    │   ├─ Evaluate coverage (0-100)
    │   ├─ Evaluate confidence (0-100)
    │   ├─ Identify missing aspects
    │   └─ Decide: Continue or refine?
    │
    └─ STEP 3: Query Refinement (if needed)
        ├─ Analyze missing aspects
        ├─ Generate new keywords
        └─ Retry STEP 1 with refined query
    │
    ▼
STEP 4: Answer Generation - GENERATOR
    ├─ Send query + context to LLM
    ├─ LLM generates answer
    ├─ Add source citations
    └─ Return grounded response

Grounded Answer with Citations
    │
    ▼
User Interface
└─ Display answer + metadata
```

## 🎯 Agent State Machine

```
                    ┌─────────────┐
                    │   INITIAL   │
                    │   State     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │   RETRIEVING    │
                    │  (Tool: retrieve)│
                    └──────┬──────────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │   REFLECTING    │
                    │  (Critic eval)  │
                    └──────┬──────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    High Quality    Medium Quality    Low Quality
    Docs Found      Docs or Need      Docs
           │         Refinement          │
           │          │                  │
        (Skip)   (Iterate)         (Stop/Empty)
           │          │                  │
           └────┬─────┘                  │
                │                        │
                ▼                        │
        ┌──────────────┐                 │
        │  ANALYZING   │◄────────────────┘
        │              │
        └──────┬───────┘
               │
               ▼
        ┌──────────────────┐
        │   GENERATING     │
        │  (LLM Answer)    │
        └──────┬───────────┘
               │
               ▼
        ┌──────────────────┐
        │    COMPLETE      │
        │  (Return Result) │
        └──────────────────┘
```

## 📦 Component Details

### 1. DocumentProcessor
- **Input**: Directory with documents
- **Processing**: Load → Chunk → Metadata
- **Output**: List of chunks with metadata

```python
chunks = [
    {
        "content": "...",        # Chunk text
        "source": "file.txt",    # Source document
        "chunk_id": 0,           # Chunk index
        "metadata": {
            "source": "file.txt",
            "chunk_index": 0,
            "total_chunks": 50
        }
    },
    ...
]
```

### 2. EmbeddingManager
- **API**: Azure OpenAI Service
- **Model**: `text-embedding-ada-002`
- **Output**: 1536-dimensional vectors
- **Batch Processing**: Handles multiple texts efficiently

### 3. FAISSVectorStore
- **Index Type**: IndexFlatL2 (exact similarity)
- **Storage**: 
  - `faiss.index`: Vector index
  - `metadata.pkl`: Document metadata
- **Operations**:
  - Add documents and embeddings
  - Search by similarity
  - Persist to disk

### 4. RAGRetriever
- **Input**: Query string
- **Processing**:
  1. Embed query
  2. Search FAISS
  3. Filter by threshold
  4. Return ranked results
- **Output**: List of (content, similarity, source) tuples

### 5. AgenticRAG
- **States**: INITIAL → RETRIEVING → REFLECTING → GENERATING → COMPLETE
- **Tool Calls**: Retriever as callable tool
- **Critic**: Evaluates document relevance
- **Generator**: Creates grounded answers
- **Multi-iteration**: Refines queries if needed

## 🔀 Execution Flow Example

```
Query: "What is the difference between supervised and unsupervised learning?"

Step 1: Generate Query Embedding
   Input: "What is the difference between..."
   API Call: Azure OpenAI Embeddings API
   Output: [0.12, -0.45, 0.78, ...]  (1536-dim vector)

Step 2: Retrieve from FAISS
   Input: Query embedding
   Search: L2 distance
   Results: Top-5 documents
   
Step 3: Critic Evaluation
   Relevance: 85% ✓
   Coverage: 90% ✓
   Confidence: 85% ✓
   → No need for refinement
   
Step 4: Generate Answer
   Input: Query + Retrieved context
   LLM: Azure OpenAI GPT-4o-mini
   Output: Well-structured answer with citations

Step 5: Return to User
   Answer: "Supervised learning uses labeled data..."
   Confidence: 85%
   Iterations: 1
   Sources: [ml_fundamentals.txt (Document 1), ...]
```

## 💾 Persistence and Caching

```
First Run:
┌────────────┐    ┌──────────────┐    ┌────────────┐
│ Documents  │───▶│ Embeddings   │───▶│   FAISS    │
│ Folder     │    │ Generation   │    │   Store    │ ──▶ Disk
└────────────┘    └──────────────┘    └────────────┘
  (1-2 min)                                 │
                                            │save()
                                            ▼
                                    ┌─────────────────┐
                                    │ faiss.index     │
                                    │ metadata.pkl    │
                                    └─────────────────┘


Subsequent Runs:
                                    ┌─────────────────┐
                                    │ faiss.index     │
                                    │ metadata.pkl    │
                                    └────────┬────────┘
                                             │load()
                                             ▼
                                    ┌─────────────────┐
                                    │   FAISS Store   │
                                    │   (in memory)   │
                                    └────────┬────────┘
                                             │
                                    Ready for queries!
                                    (instant access)
```

## 📊 Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Document Processing | ~1s per document | Depends on size |
| Embedding Generation | ~2s per 50 chunks | API call limited |
| Vector Store Build | ~5s for 1000 chunks | FAISS indexing |
| Query Embedding | ~200ms | Single API call |
| Similarity Search | ~50ms | FAISS L2 distance |
| Critic Evaluation | ~3-5s | Additional LLM call |
| Answer Generation | ~5-10s | LLM inference |
| **Total Latency** | **~8-20s** | Per query (1 iteration) |

## 🎓 Learning Components

### Document Chunking
- Fixed-size chunks: 1000 characters
- Overlapping windows: 200 character overlap
- Sentence-aware breaking: Try to break at periods/newlines

### Embedding Quality
- Dense representations: 1536 dimensions (text-embedding-ada-002)
- Semantic meaning: Understands context
- Query-document alignment: Cosine/L2 similarity

### Retrieval Ranking
- L2 Distance: `distance = sqrt(sum((a-b)^2))`
- Similarity: `similarity = 1 / (1 + distance)`
- Threshold: Only return if similarity > 0.3

### Answer Grounding
- Retrieved documents kept in context
- LLM instructed to cite sources
- Explicit acknowledgment of missing info
- Minimized hallucination

## 🔐 Safety and Hallucination Prevention

1. **Retrieval Grounding**: Answers based on real documents
2. **Threshold Filtering**: Ignore low-relevance results
3. **Explicit Uncertainty**: State when unsure
4. **Source Attribution**: Always cite sources
5. **Critic Evaluation**: Validate document quality
6. **Iterative Refinement**: Multiple attempts if needed
7. **Confidence Scoring**: Show how confident system is
