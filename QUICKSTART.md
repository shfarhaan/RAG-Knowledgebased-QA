# Quick Start Guide - Agentic RAG System

## ⚡ 5-Minute Setup

### 1. Get Your Azure OpenAI Credentials
- You need:
  - Azure OpenAI API Key
  - Azure OpenAI Endpoint URL
- These are provided by your Azure OpenAI resource

### 2. Install Dependencies
```bash
# Navigate to project directory
cd "f:\Selise Assessment"

# OPTION A: Automatic installer (recommended for Windows)
install.bat

# OPTION B: Manual installation
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Verify installation
python setup.py
```

> **Note**: Import warnings in VS Code are normal before installation and will disappear after running the installer.

### 3. Configure Azure OpenAI Credentials
Create a `.env` file in the project root:
```
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com
```

### 4. Add Your Documents
- Place `.txt` or `.md` files in the `documents/` folder
- Or use the pre-included samples

### 5. Choose Your Interface

#### Option A: Web UI (Recommended)
```bash
streamlit run app.py
```
Then:
1. Go to "Process Documents" tab
2. Click "Process Documents"
3. Go to "Ask Questions" tab
4. Start asking questions!

#### Option B: Command Line
```bash
# Process documents
python cli.py process

# Start chatting
python cli.py chat
```

#### Option C: Jupyter Notebook
```bash
jupyter notebook notebook_example.ipynb
```

---

## 🎯 What Can You Do?

### With Sample Documents:
- "What is machine learning?"
- "Explain the difference between supervised and unsupervised learning"
- "What are advantages of RAG systems?"
- "How do embeddings work?"

### With Your Documents:
- Ask any questions about your domain knowledge
- Get grounded answers with source citations
- See confidence scores and reasoning steps

---

## 🏗️ System Architecture

```
Your Question
      ↓
┌──────────────────────────────┐
│     Agentic RAG Agent        │
│ ┌─────────────┐              │
│ │ Retriever   │ ← Get docs   │
│ ├─────────────┤              │
│ │ Critic      │ ← Evaluate   │
│ ├─────────────┤              │
│ │ Generator   │ ← Answer     │
│ └─────────────┘              │
└──────────────────────────────┘
      ↓
Grounded Answer with Citations
```

---

## 🔧 Configuration (Optional)

Edit `src/config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| CHUNK_SIZE | 1000 | Size of document chunks |
| CHUNK_OVERLAP | 200 | Overlap between chunks |
| TOP_K_DOCUMENTS | 5 | Documents to retrieve |
| TEMPERATURE | 0.3 | Response randomness (0=deterministic) |
| SIMILARITY_THRESHOLD | 0.3 | Minimum relevance score |

---

## ❓ Troubleshooting

### Import warnings in VS Code (e.g., "Import 'streamlit' could not be resolved")
- **This is normal before installation!**
- Run: `install.bat` or `pip install -r requirements.txt`
- Warnings will disappear after installation
- Verify with: `python setup.py`

### "Azure OpenAI credentials not found"
- Make sure `.env` file exists in project root
- Check that both `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are set
- Ensure there are no typos in variable names
- Restart the application

### "No relevant documents found"
- Make sure you processed documents first
- Upload documents to `documents/` folder
- Click "Process Documents" in the UI

### "Vector store not found"
- Process documents using "Process Documents" button
- Wait for embeddings to generate (takes 1-2 minutes)

### Can't run `streamlit run app.py`?
- Ensure you're in the project directory
- Check virtual environment is activated
- Run `pip install streamlit` to ensure it's installed

---

## 📚 Next Steps

1. **Explore the System**
   - Process sample documents
   - Ask various questions
   - Check confidence scores

2. **Add Your Documents**
   - Replace sample documents with your content
   - Process with custom chunk sizes
   - Test question-answering

3. **Customize**
   - Adjust temperature for more/less deterministic responses
   - Change chunk size for different content
   - Modify TOP_K_DOCUMENTS for retrieval depth

4. **Deploy**
   - Deploy Streamlit app to Streamlit Cloud
   - Set secrets in cloud
   - Share the link with others

---

## 📖 Key Concepts

### Chunking
Breaking large documents into smaller pieces for better retrieval relevance.

### Embeddings
Converting text into numerical vectors that capture semantic meaning.

### Vector Store
Database that stores embeddings for fast similarity search (FAISS in our case).

### Retriever
Finds relevant documents based on query similarity.

### Critic
Evaluates if retrieved documents are sufficient to answer the question.

### Generator
LLM that creates the final answer grounded in retrieved documents.

### Agentic RAG
Uses tool-calling, reflection, and iterative refinement for better answers.

---

## 📞 Support

For issues:
1. Check the main `README.md` for detailed documentation
2. Review logs in the terminal
3. Ensure API key is set correctly
4. Verify documents are in `documents/` folder

---

## 🚀 You're Ready!

Now go ahead and:
```bash
streamlit run app.py
```

Happy questioning! 🎉
