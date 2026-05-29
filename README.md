# Hybrid Procedural RAG — React + FastAPI

```
project/
├── backend/
│   ├── main.py           ← FastAPI server (RAG + LLM + image logic)
│   ├── ingest.py         ← Standalone ingestion (.txt .pdf .docx .md .csv)
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        └── App.jsx       ← React UI (typing animation, images, upload panel)
```

---

## What's new

| Feature | Where |
|---|---|
| Three-dot typing animation while waiting for reply | `App.jsx` → `TypingIndicator` |
| `extract_images_with_context` — image URLs + surrounding text | `main.py` |
| `render_steps_with_semantic_images` — cosine-sim image→step matching | `main.py` |
| Fallback log `"No procedural results found -> fallback to hybrid search"` | `main.py` → `_retrieve` |
| Non-procedural answers focus strictly on the user's question | `main.py` → `_generate_focused_answer` |
| Images displayed inline under each bot message | `App.jsx` → `ImageGallery` |
| Separate `ingest.py` for document ingestion | `ingest.py` |
| Multi-file-type upload: `.txt .pdf .docx .md .csv` | `ingest.py` + `/upload` endpoint |
| Upload panel with drag-and-drop and indexed sources list | `App.jsx` → `UploadPanel` |
| `/indexed-sources` endpoint to list all indexed files | `main.py` |

---

## 1. Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# For PDF support
pip install pdfplumber

# For DOCX support
pip install python-docx

# Make sure Ollama is running
ollama serve
ollama pull llama3:instruct

# Start the API
uvicorn main:app --reload --port 8000
```

---

## 2. Ingest documents (CLI)

```bash
# Single file (any supported type)
python ingest.py file path/to/manual.pdf
python ingest.py file path/to/guide.docx
python ingest.py file path/to/notes.md

# Whole directory
python ingest.py dir path/to/docs/

# List what's already indexed
python ingest.py list
```

You can also upload from the UI via the **📂 Upload document** button in the sidebar.

---

## 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

App available at http://localhost:3000

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /session | Create a new conversation session |
| POST | /chat | Send a message → `{reply, state, images}` |
| POST | /upload | Upload a file (.txt/.pdf/.docx/.md/.csv) |
| GET | /indexed-sources | List all indexed source filenames |
| GET | /health | Health check + doc count |

### Chat response now includes images

```json
{
  "reply": "Here is how to connect the device…",
  "state": "awaiting_step_confirmation",
  "images": [
    "https://example.com/step1-diagram.png"
  ]
}
```

---

## Supported file types

| Extension | Library needed |
|---|---|
| `.txt` | none |
| `.md` | none |
| `.csv` | none |
| `.pdf` | `pip install pdfplumber` |
| `.docx` | `pip install python-docx` |
