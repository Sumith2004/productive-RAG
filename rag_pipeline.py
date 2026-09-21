"""Shared RAG pipeline logic used by 01_ingest.ipynb, 02_query.ipynb, and app.py.

Partition PDF -> chunk by title -> summarize (Groq, text-only) -> embed (HuggingFace) -> store (Chroma).
Then: retrieve top-k chunks by cosine similarity -> generate answer (Groq).
"""

import json
import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_llm():
    return ChatGroq(model=GROQ_MODEL, temperature=0)


# --- Step 1: Partition document (PDF or DOCX) into atomic elements (text, tables, images) ---

def partition_document(file_path: str):
    return partition(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )


# --- Step 2: Group atomic elements into chunks by title ---

def create_chunks_by_title(elements):
    return chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )


def separate_content_types(chunk):
    content_data = {"text": chunk.text, "tables": [], "images": [], "types": ["text"]}

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


def create_ai_enhanced_summary(llm, text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content (text-only Groq model - no vision model available)"""
    try:
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """

        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        if images:
            prompt_text += f"NOTE: This chunk also contains {len(images)} image(s) that could not be visually analyzed.\n\n"

        prompt_text += """
        YOUR TASK:
        Generate a comprehensive, searchable description that covers:

        1. Key facts, numbers, and data points from text and tables
        2. Main topics and concepts discussed
        3. Questions this content could answer
        4. Alternative search terms users might use

        Make it detailed and searchable - prioritize findability over brevity.

        SEARCHABLE DESCRIPTION:"""

        response = llm.invoke([HumanMessage(content=prompt_text)])
        return response.content

    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


# --- Step 3: Summarize chunks and wrap into LangChain Documents ---

def summarise_chunks(llm, chunks, progress_callback=None):
    langchain_documents = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback((i + 1) / total_chunks, f"Summarizing chunk {i+1}/{total_chunks}")

        content_data = separate_content_types(chunk)

        if content_data["tables"] or content_data["images"]:
            enhanced_content = create_ai_enhanced_summary(
                llm, content_data["text"], content_data["tables"], content_data["images"]
            )
        else:
            enhanced_content = content_data["text"]

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "types": ",".join(content_data["types"]),
                "original_content": json.dumps(
                    {
                        "raw_text": content_data["text"],
                        "tables_html": content_data["tables"],
                        "images_base64": content_data["images"],
                    }
                ),
            },
        )
        langchain_documents.append(doc)

    return langchain_documents


# --- Step 4: Embed documents and store in Chroma ---

def build_vector_store(documents, embedding_model, persist_directory=None):
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vector_store(persist_directory, embedding_model):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


# --- Full ingestion pipeline: partition -> chunk -> summarize -> embed & store ---

def run_ingestion_pipeline(pdf_path, llm, embedding_model, persist_directory=None, progress_callback=None):
    if progress_callback:
        progress_callback(0.05, "Partitioning PDF...")
    elements = partition_document(pdf_path)

    if progress_callback:
        progress_callback(0.3, "Chunking by title...")
    chunks = create_chunks_by_title(elements)

    documents = summarise_chunks(llm, chunks, progress_callback=progress_callback)

    if progress_callback:
        progress_callback(0.95, "Building vector store...")
    db = build_vector_store(documents, embedding_model, persist_directory)

    return db, documents


# --- Query pipeline: retrieve top-k chunks, generate answer ---

def retrieve_chunks(db, query, k=3):
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def generate_final_answer(llm, chunks, query):
    prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""

    for i, chunk in enumerate(chunks):
        prompt_text += f"--- Document {i+1} ---\n"

        if "original_content" in chunk.metadata:
            original_data = json.loads(chunk.metadata["original_content"])

            raw_text = original_data.get("raw_text", "")
            if raw_text:
                prompt_text += f"TEXT:\n{raw_text}\n\n"

            tables_html = original_data.get("tables_html", [])
            if tables_html:
                prompt_text += "TABLES:\n"
                for j, table in enumerate(tables_html):
                    prompt_text += f"Table {j+1}:\n{table}\n\n"

        prompt_text += "\n"

    prompt_text += """
Please provide a clear, comprehensive answer using the text and tables above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

If your answer includes any mathematical notation (equations, subscripts, symbols), format it as LaTeX wrapped in single dollar signs, e.g. $d_k = d_{model}/h$.

ANSWER:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        return response.content
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."
