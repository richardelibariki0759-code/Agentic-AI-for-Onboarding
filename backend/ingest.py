"""
ingest.py  —  Standalone document ingestion for Hybrid Procedural RAG
Supports: .txt, .pdf, .docx, .md, .csv
Run directly:  python ingest.py path/to/file.pdf
Or import and call ingest_file(path) from another module.
"""

import argparse
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "txt_docs"
SIMILARITY_THRESHOLD = 0.72
MAX_CHUNK_SENTENCES = 8
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".csv"}

# ─────────────────────────────────────────────────────────────────────────────
# INIT (module-level singletons so the model loads once when imported)
# ─────────────────────────────────────────────────────────────────────────────

print("[ingest] Loading embedding model …")
_embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)

print(f"[ingest] ChromaDB ready at '{CHROMA_PATH}'  "
      f"(existing docs: {_collection.count()})")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION  (per file type)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_md(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # strip markdown syntax but keep image URLs
    text = re.sub(r"!\[.*?\]\((.*?)\)", r"\1", text)  # ![alt](url) → url
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)         # [text](url) → ''
    text = re.sub(r"[#*`>_~|]+", " ", text)
    return text


def _extract_pdf(path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
            return "\n\n".join(pages)
    except ImportError:
        raise RuntimeError(
            "pdfplumber is required for PDF ingestion.\n"
            "Install it with:  pip install pdfplumber"
        )


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise RuntimeError(
            "python-docx is required for .docx ingestion.\n"
            "Install it with:  pip install python-docx"
        )


def _extract_csv(path: Path) -> str:
    import csv
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(" | ".join(row))
    return "\n".join(rows)


EXTRACTORS = {
    ".txt":  _extract_txt,
    ".md":   _extract_md,
    ".pdf":  _extract_pdf,
    ".docx": _extract_docx,
    ".csv":  _extract_csv,
}


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in EXTRACTORS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return EXTRACTORS[ext](path)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_step_number(text: str) -> Optional[int]:
    patterns = [
        r"step\s+(\d+)",
        r"^(\d+)\.",
        r"\n(\d+)\.",
    ]
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def semantic_chunk(
    text: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    max_chunk_sentences: int = MAX_CHUNK_SENTENCES,
) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []

    embeds = _embedding_model.encode(
        [f"passage: {s}" for s in sentences],
        normalize_embeddings=True,
    )

    chunks: list[str] = []
    current = [sentences[0]]
    cur_emb = embeds[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([cur_emb], [embeds[i]])[0][0]
        if sim >= similarity_threshold and len(current) < max_chunk_sentences:
            current.append(sentences[i])
            cur_emb = np.mean(embeds[i - len(current) + 1: i + 1], axis=0)
        else:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            cur_emb = embeds[i]

    if current:
        chunks.append(" ".join(current))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    return _embedding_model.encode(
        f"passage: {text}", normalize_embeddings=True
    ).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# CORE INGEST
# ─────────────────────────────────────────────────────────────────────────────

def ingest_file(path: str | Path) -> dict:
    """
    Extract, chunk, embed, and store a document in ChromaDB.

    Returns:
        {"source": filename, "chunks_added": N}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"[ingest] Extracting text from '{path.name}' …")
    raw = extract_text(path)
    text = _clean(raw)

    print(f"[ingest] Chunking …")
    chunks = semantic_chunk(text)
    print(f"[ingest] {len(chunks)} chunks created")

    added = 0
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        step_number = _detect_step_number(chunk)
        emb = _embed(chunk)
        doc_id = str(uuid.uuid4())

        _collection.add(
            documents=[chunk],
            embeddings=[emb],
            metadatas=[
                {
                    "source": path.name,
                    "chunk_index": i,
                    "step_number": step_number if step_number is not None else -1,
                    "is_procedural": step_number is not None,
                    "file_type": path.suffix.lower(),
                }
            ],
            ids=[doc_id],
        )
        added += 1

    print(f"[ingest] ✅ Stored {added} chunks from '{path.name}'")
    return {"source": path.name, "chunks_added": added}


def ingest_directory(dir_path: str | Path) -> list[dict]:
    """Ingest all supported files in a directory (non-recursive)."""
    dir_path = Path(dir_path)
    results = []
    for p in sorted(dir_path.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file():
            try:
                results.append(ingest_file(p))
            except Exception as e:
                print(f"[ingest] ⚠️  Skipping '{p.name}': {e}")
    return results


def list_indexed_sources() -> list[str]:
    """Return unique source filenames already in ChromaDB."""
    data = _collection.get(include=["metadatas"])
    sources = {m.get("source", "unknown") for m in data.get("metadatas", [])}
    return sorted(sources)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest documents into ChromaDB for the Hybrid Procedural RAG system."
    )
    subparsers = parser.add_subparsers(dest="command")

    # ingest a file
    p_file = subparsers.add_parser("file", help="Ingest a single file")
    p_file.add_argument("path", help="Path to the file to ingest")

    # ingest a directory
    p_dir = subparsers.add_parser("dir", help="Ingest all supported files in a directory")
    p_dir.add_argument("path", help="Path to the directory")

    # list indexed sources
    subparsers.add_parser("list", help="List all indexed source files")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.command == "file":
        result = ingest_file(args.path)
        print(f"\nDone: {result}")

    elif args.command == "dir":
        results = ingest_directory(args.path)
        print(f"\nDone: {len(results)} files ingested")
        for r in results:
            print(f"  • {r['source']} → {r['chunks_added']} chunks")

    elif args.command == "list":
        sources = list_indexed_sources()
        if sources:
            print(f"\nIndexed sources ({len(sources)}):")
            for s in sources:
                print(f"  • {s}")
        else:
            print("\nNo documents indexed yet.")

    else:
        print("Usage:  python ingest.py file <path>")
        print("        python ingest.py dir  <path>")
        print("        python ingest.py list")
        sys.exit(1)
