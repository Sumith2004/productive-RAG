# Productive RAG

Streamlit chatbot for document Q&A. Upload PDF/DOCX, get RAG-backed answers over text, tables, and images.

## Pipeline

1. Partition document (`unstructured`, hi-res strategy) into text/table/image elements
2. Chunk by title
3. Summarize mixed-content chunks (Groq LLM) for better searchability
4. Embed (`sentence-transformers/all-MiniLM-L6-v2`) and store in Chroma
5. Query: retrieve top-k chunks by cosine similarity -> generate answer (Groq)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
```

## Run

```bash
streamlit run app.py
```

Create a chatbot, upload a PDF or DOCX, ask questions. Chatbots persist across sessions (`sessions/` dir, manifest-tracked); Chroma DB stored per session.

## Files

- `app.py` — Streamlit UI
- `rag_pipeline.py` — shared ingestion/query logic (used by app + notebooks)
- `01_ingest.ipynb`, `02_query.ipynb` — notebook versions of the pipeline
