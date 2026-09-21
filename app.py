import base64
import json
import os
import re
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

import rag_pipeline

load_dotenv()

SESSIONS_DIR = "sessions"
MANIFEST_PATH = os.path.join(SESSIONS_DIR, "manifest.json")


# --- Session persistence (save/load created chatbots to/from disk) ---

def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest):
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def make_session_slug(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower()) + "_" + uuid.uuid4().hex[:8]


def save_session(slug, name, filename, documents, chroma_dir):
    session_dir = os.path.join(SESSIONS_DIR, slug)
    os.makedirs(session_dir, exist_ok=True)

    docs_data = [{"page_content": d.page_content, "metadata": d.metadata} for d in documents]
    with open(os.path.join(session_dir, "documents.json"), "w", encoding="utf-8") as f:
        json.dump(docs_data, f, indent=2)

    manifest = load_manifest()
    manifest.append(
        {
            "slug": slug,
            "name": name,
            "filename": filename,
            "chroma_dir": chroma_dir,
        }
    )
    save_manifest(manifest)


def load_session_documents(slug):
    path = os.path.join(SESSIONS_DIR, slug, "documents.json")
    with open(path, "r", encoding="utf-8") as f:
        docs_data = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs_data]


@st.cache_resource
def load_embedding_model():
    return rag_pipeline.get_embedding_model()


@st.cache_resource
def load_llm():
    return rag_pipeline.get_llm()


def run_ingestion(uploaded_file, llm, embedding_model, persist_directory, progress_callback=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    db, documents = rag_pipeline.run_ingestion_pipeline(
        tmp_path, llm, embedding_model, persist_directory, progress_callback
    )

    os.remove(tmp_path)
    return db, documents


# --- Document viewer (chunk inspector) ---

@st.dialog("Uploaded Document", width="large")
def show_document_viewer():
    documents = st.session_state.documents
    filename = st.session_state.filename

    st.markdown(f"**{filename}** — Processing Pipeline")

    tab_chunking, tab_summarisation, tab_vectorization, tab_view_chunks = st.tabs(
        ["Chunking", "Summarisation", "Vectorization & Storage", "View Chunks"]
    )

    total = len(documents)
    text_count = sum(1 for d in documents if d.metadata.get("types") == "text")
    table_count = sum(1 for d in documents if "table" in d.metadata.get("types", ""))
    image_count = sum(1 for d in documents if "image" in d.metadata.get("types", ""))

    with tab_chunking:
        st.metric("Total chunks", total)
        col1, col2, col3 = st.columns(3)
        col1.metric("Text-only chunks", text_count)
        col2.metric("Chunks with tables", table_count)
        col3.metric("Chunks with images", image_count)

    with tab_summarisation:
        summarized = sum(
            1 for d in documents if "table" in d.metadata.get("types", "") or "image" in d.metadata.get("types", "")
        )
        st.metric("Chunks AI-summarized", summarized)
        st.caption("Chunks with tables or images get an AI-enhanced searchable summary; plain text chunks use raw text as-is.")

    with tab_vectorization:
        st.metric("Vectors stored", total)
        st.caption(f"Embedding model: {rag_pipeline.EMBEDDING_MODEL_NAME}")
        st.caption("Vector store: Chroma (persisted per session)")

    with tab_view_chunks:
        col_filter, col_search = st.columns([1, 2])
        with col_filter:
            type_filter = st.radio("Filter", ["All", "Text", "Image", "Table"], horizontal=True, label_visibility="collapsed")
        with col_search:
            search_term = st.text_input("Search chunks...", label_visibility="collapsed", placeholder="Search chunks...")

        filtered = []
        for i, doc in enumerate(documents):
            types = doc.metadata.get("types", "")
            if type_filter == "Text" and types != "text":
                continue
            if type_filter == "Image" and "image" not in types:
                continue
            if type_filter == "Table" and "table" not in types:
                continue
            if search_term and search_term.lower() not in doc.page_content.lower():
                continue
            filtered.append((i, doc))

        st.caption(f"{len(filtered)} of {total} chunks")

        if "selected_chunk" not in st.session_state:
            st.session_state.selected_chunk = filtered[0][0] if filtered else None

        list_col, detail_col = st.columns([1, 1])

        with list_col:
            for i, doc in filtered:
                types = doc.metadata.get("types", "text")
                label = f"[{types}] Chunk {i+1}: {doc.page_content[:60]}..."
                if st.button(label, key=f"chunk_{i}", use_container_width=True):
                    st.session_state.selected_chunk = i

        with detail_col:
            if st.session_state.selected_chunk is not None:
                selected_doc = documents[st.session_state.selected_chunk]
                original_data = json.loads(selected_doc.metadata.get("original_content", "{}"))

                summary_tab, original_tab = st.tabs(["Summary", "Original"])

                with summary_tab:
                    st.markdown(selected_doc.page_content)

                with original_tab:
                    st.text_area("Raw text", original_data.get("raw_text", ""), height=200, disabled=True)

                    tables_html = original_data.get("tables_html", [])
                    if tables_html:
                        st.markdown("**Tables**")
                        for t in tables_html:
                            st.markdown(t, unsafe_allow_html=True)

                    images_base64 = original_data.get("images_base64", [])
                    if images_base64:
                        st.markdown("**Images**")
                        for img_b64 in images_base64:
                            st.image(base64.b64decode(img_b64))
            else:
                st.caption("No chunks match the current filter/search.")


# --- UI ---

st.set_page_config(page_title="RAG Chatbot")
st.title("RAG Chatbot")

if "stage" not in st.session_state:
    st.session_state.stage = "start"  # start -> naming -> uploading -> ready
if "name" not in st.session_state:
    st.session_state.name = ""
if "slug" not in st.session_state:
    st.session_state.slug = ""
if "db" not in st.session_state:
    st.session_state.db = None
if "documents" not in st.session_state:
    st.session_state.documents = None
if "filename" not in st.session_state:
    st.session_state.filename = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

llm = load_llm()
embedding_model = load_embedding_model()

if st.session_state.stage == "start":
    existing_sessions = load_manifest()

    if existing_sessions:
        st.subheader("Your chatbots")
        for session in existing_sessions:
            if st.button(session["name"], key=f"open_{session['slug']}", use_container_width=True):
                db = rag_pipeline.load_vector_store(session["chroma_dir"], embedding_model)
                st.session_state.db = db
                st.session_state.documents = load_session_documents(session["slug"])
                st.session_state.filename = session["filename"]
                st.session_state.name = session["name"]
                st.session_state.slug = session["slug"]
                st.session_state.messages = []
                st.session_state.stage = "ready"
                st.rerun()
        st.divider()

    if st.button("Create"):
        st.session_state.stage = "naming"
        st.rerun()

elif st.session_state.stage == "naming":
    name = st.text_input("Name this chatbot")
    if st.button("Enter", disabled=not name.strip()):
        st.session_state.name = name.strip()
        st.session_state.stage = "uploading"
        st.rerun()

elif st.session_state.stage == "uploading":
    st.subheader(f"{st.session_state.name}: upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("Upload", disabled=uploaded_file is None):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def progress_callback(fraction, message):
            progress_bar.progress(fraction)
            status_text.text(message)

        slug = make_session_slug(st.session_state.name)
        chroma_dir = os.path.join(SESSIONS_DIR, slug, "chroma_db")

        db, documents = run_ingestion(uploaded_file, llm, embedding_model, chroma_dir, progress_callback)
        save_session(slug, st.session_state.name, uploaded_file.name, documents, chroma_dir)

        st.session_state.db = db
        st.session_state.documents = documents
        st.session_state.filename = uploaded_file.name
        st.session_state.slug = slug
        progress_bar.progress(1.0)
        status_text.text("Done")
        st.session_state.stage = "ready"
        st.rerun()

else:
    col_title, col_doc_btn, col_reset = st.columns([3, 2, 1])
    with col_title:
        st.subheader(st.session_state.name)
    with col_doc_btn:
        if st.button("📄 Uploaded Document"):
            show_document_viewer()
    with col_reset:
        if st.button("Back to list"):
            st.session_state.stage = "start"
            st.session_state.name = ""
            st.session_state.slug = ""
            st.session_state.db = None
            st.session_state.documents = None
            st.session_state.filename = ""
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about the document")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chunks = rag_pipeline.retrieve_chunks(st.session_state.db, user_query)
                answer = rag_pipeline.generate_final_answer(llm, chunks, user_query)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
