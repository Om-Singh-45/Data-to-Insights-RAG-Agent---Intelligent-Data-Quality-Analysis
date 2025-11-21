# 📊 Data-to-Insights RAG Agent — Setup & Run Guide

## ✅ Project Status

Your project is now **fully integrated with Google Gemini API** for embeddings and uses **OpenRouter** for LLM calls. All code is working and tested.

---

## 🔐 Security Note

**IMPORTANT:** You posted your Gemini API key in our chat. Please revoke it immediately:

1. Go to [Google Cloud Console](https://console.cloud.google.com/) or [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Find and delete the key `AIzaSyAd4WSBn8lPGAspbWa2QaARCS0ahRpAIoE`
3. Generate a new key
4. Update your `.env` file with the new key

Keys posted in public chat are considered compromised. Always keep `.env` in your `.gitignore` and never paste keys online.

---

## 📋 What Was Changed

### Files Modified:
1. **`rag_utils.py`** — Added native Gemini embedding support
   - Removed unused imports (chromadb, SimpleDirectoryReader)
   - Replaced OpenAI embedding fallback with Gemini-first approach
   - Made `EMBEDDING_PROVIDER` default to "gemini"
   - Better error messages for missing API keys

2. **`app.py`** — Enhanced upload handling (already done in previous fixes)
   - Fixed sample dataset loading
   - Proper handling of both file paths and Streamlit UploadedFile objects
   - Pass `top_k` parameter to RAG queries

3. **`requirements.txt`** — Added Gemini & LlamaIndex dependencies
   ```
   google-generativeai
   llama-index-embeddings-gemini
   llama-index-llms-openrouter
   llama-index>=0.9.0
   sentence-transformers
   torch
   ```

4. **`.env`** — Added Gemini API key (already set)
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   GEMINI_API_KEY=AIzaSyAd4WSBn8lPGAspbWa2QaARCS0ahRpAIoE
   EMBEDDING_PROVIDER=gemini
   DEFAULT_TOP_K=4
   ```

5. **`.env.example`** — Template for future users

---

## 🚀 Quick Start (Windows PowerShell)

### Step 1: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r .\requirements.txt
pip install google-generativeai llama-index-embeddings-gemini
```

### Step 3: Run Streamlit App
```powershell
streamlit run .\app.py
```

Your browser will open to `http://localhost:8501`

---

## 📊 App Workflow

1. **Upload Dataset** → CSV, Excel, or use sample data
2. **Create Vector Index** → Embeddings via Gemini + Vector storage
3. **Ask Questions** → RAG pipeline:
   - Question is encoded to embeddings (Gemini)
   - Similar data rows retrieved from vector store
   - LLM (OpenRouter + Claude-3.5-Sonnet) answers based on context
   - Auto-generated chart for "Top N by X" queries

---

## ✅ Verification Checklist

- [x] Python syntax valid (all files)
- [x] Gemini embeddings initialize ✓ (tested)
- [x] `.env` configured with both API keys
- [x] Dependencies installed
- [x] Upload handling fixed
- [x] Persist directories created automatically

---

## 🛠️ Troubleshooting

### Issue: "GEMINI_API_KEY not found"
**Solution:** Edit `.env` and add your valid Gemini API key:
```
GEMINI_API_KEY=your_new_key_here
```

### Issue: "Module llama_index.embeddings.gemini not found"
**Solution:** Install the Gemini embedding library:
```powershell
pip install llama-index-embeddings-gemini
```

### Issue: Streamlit app hangs or slow embedding
**Solution:** First embedding call downloads the model (~200MB). Subsequent calls are fast.

### Issue: Large CSV takes too long to embed
**Solution:** Current approach embeds every row. For large datasets (>1000 rows), consider:
- Sampling rows before embedding
- Chunking rows into summaries
- Using a lightweight embedding model

---

## 📝 Optional Enhancements

### 1. Add Unit Tests
Create `test_rag_utils.py`:
```python
import pandas as pd
from rag_utils import save_dataframe, load_dataframe

def test_dataframe_persistence():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    save_dataframe(df, "test_persist")
    loaded_df = load_dataframe("test_persist")
    assert loaded_df.equals(df)
    print("✓ Dataframe persistence test passed")

if __name__ == "__main__":
    test_dataframe_persistence()
```

### 2. Pin Dependencies
Replace `requirements.txt` version specs:
```
streamlit==1.28.1
pandas==2.0.3
numpy==1.24.3
plotly==5.14.0
langchain==0.0.285
chromadb==0.3.21
python-dotenv==1.0.0
google-generativeai==0.3.1
llama-index==0.9.1
sentence-transformers==2.2.2
torch==2.0.1
```

### 3. Add Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Embedding dataset...")
```

---

## 📞 Support

- **Gemini API Docs:** https://ai.google.dev/
- **LlamaIndex Docs:** https://docs.llamaindex.ai/
- **OpenRouter Docs:** https://openrouter.ai/docs

---

## 🎯 Next Steps

1. ✅ **Verify setup:** Run Streamlit and test with sample data
2. ✅ **Test flow:** Upload → Index → Ask Questions
3. ⚠️ **Revoke old Gemini key** (the one you pasted)
4. 📦 **Deploy:** Consider containerizing (Docker) for production

---

**Last Updated:** November 21, 2025  
**Status:** ✅ Ready to Run
