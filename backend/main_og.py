"""
Hybrid + Procedural RAG — FastAPI Backend
Includes:
  - extract_images_with_context  (image URLs + surrounding context)
  - render_steps_with_semantic_images  (cosine-sim image→step matching)
  - Procedural fallback logging: "No procedural results found -> fallback to hybrid search"
  - Focused non-procedural answers (only address the user's exact question)
  - Multi-file-type upload endpoint (delegates to ingest.py)
  - Session-based conversational state machine
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import chromadb
import re
import numpy as np
import ollama
import uuid
import tempfile
import os

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# Import ingest helpers (same package)
from ingest import (
    ingest_file,
    semantic_chunk,
    SUPPORTED_EXTENSIONS,
)

# =========================================================
# APP
# =========================================================

app = FastAPI(title="Hybrid Procedural RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# MODELS & DB
# =========================================================

embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="txt_docs")

_docs: list[str] = []
_doc_embeddings: dict[str, list] = {}
_doc_metadata: dict[str, dict] = {}
_bm25: Optional[BM25Okapi] = None

sessions: dict[str, dict] = {}


def _rebuild_bm25():
    global _docs, _doc_embeddings, _doc_metadata, _bm25
    data = collection.get(include=["documents", "embeddings", "metadatas"])
    docs = data.get("documents", [])
    embeddings = data.get("embeddings", [])
    metadatas = data.get("metadatas", [])
    _docs = docs
    for doc, emb, meta in zip(docs, embeddings, metadatas):
        _doc_embeddings[doc] = emb
        _doc_metadata[doc] = meta
    if docs:
        _bm25 = BM25Okapi([_tokenize(d) for d in docs])


@app.on_event("startup")
def startup():
    _rebuild_bm25()


# =========================================================
# CONSTANTS
# =========================================================

PROCEDURAL_KEYWORDS = [
    "how", "steps", "procedure", "workflow", "guide",
    "install", "setup", "configure", "assemble", "upload",
    "start", "run", "continue", "what next", "process",
    "tight", "fix", "attach", "connect", "place", "insert",
    "build", "create", "make", "do", "perform",
]

GARBAGE_PATTERNS = [
    r"(?i)comprehensive guide",
    r"(?i)introduction",
    r"(?i)assembl\w+ bruno.*guide",
    r"(?i)^step\s+\d+\s*:\s*(a\s+)?[\w\s]*(guide|overview|introduction)\s*\**$",
]

CASUAL_PATTERNS = [
    r"^\s*(hi|hello|hey|hiya|howdy|greetings|good\s+(morning|afternoon|evening|day))\s*[!.,]?\s*$",
    r"^\s*(ok|okay|alright|sure|fine|got it|noted|understood|cool|great|nice|thanks|thank you|thx|ty|np|no problem|you'?re welcome|yw)\s*[!.,]?\s*$",
    r"^\s*(yes|no|nope|yep|yeah|yup|nah)\s*[!.,]?\s*$",
    r"^\s*(how are you|how r u|how do you do|what's up|wassup|sup)\s*[?!.,]?\s*$",
    r"^\s*(bye|goodbye|see you|cya|later|take care)\s*[!.,]?\s*$",
]

TOPIC_EXAMPLES = {
    "bruno": [
        "Assembling the Bruno device",
        "Connecting Bruno hardware components",
        "Troubleshooting Bruno setup issues",
    ],
    "ona app": [
        "Import & Manage Trials",
        "Data Collection methods in ONA",
        "Upload & Sync Data",
    ],
    "ona": [
        "Creating forms in ONA",
        "Uploading or submitting data using ONA",
        "Managing projects in ONA",
    ],
    "data collection": [
        "Collecting field data with ONA",
        "Image data collection",
        "Submitting collected data",
        "Image Capture (Bruno)",
        "Offline Data Collection",
        "Classic Phenotyping",
    ],
}

# =========================================================
# IMAGE EXTRACTION
# =========================================================

def extract_images_with_context(text: str, window: int = 100) -> tuple[str, list[dict]]:
    """
    Extract image URLs from text along with surrounding context.
    Returns (cleaned_text_without_urls, list_of_{"url": ..., "context": ...})
    """
    pattern = r"(https?://[^\s]+\.(jpg|jpeg|png|webp))"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    results = []
    for m in matches:
        context = text[max(0, m.start() - window): m.end() + window]
        results.append({"url": m.group(), "context": context})
    clean = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return clean, results


def render_steps_with_semantic_images(
    answer: str,
    threshold: float = 0.3,
) -> list[dict]:
    """
    Split answer into steps, match images to steps semantically.
    Returns list of {"text": str, "images": [url, ...]} per step.
    """
    if not answer:
        return []

    steps = answer.split("\n")
    shown: set[str] = set()
    rendered = []

    for step in steps:
        step = step.strip()
        if not step:
            continue

        text, images = extract_images_with_context(step)
        matched_urls = []

        if images:
            step_emb = embedding_model.encode(step, normalize_embeddings=True)
            for img in images:
                if img["url"] in shown:
                    continue
                img_emb = embedding_model.encode(img["context"], normalize_embeddings=True)
                sim = float(np.dot(step_emb, img_emb))
                if sim < threshold:
                    continue
                shown.add(img["url"])
                matched_urls.append(img["url"])

        rendered.append({"text": text, "images": matched_urls})

    return rendered


# =========================================================
# HELPERS
# =========================================================

def _is_casual(text: str) -> bool:
    t = text.strip().lower()
    return any(re.match(p, t) for p in CASUAL_PATTERNS)


def _tokenize(text: str) -> list[str]:
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text.lower())
    return [t for t in text.split() if len(t) > 2]


def _get_embedding(text: str, is_query: bool = False) -> list[float]:
    prefix = "query: " if is_query else "passage: "
    emb = embedding_model.encode(prefix + text, normalize_embeddings=True)
    return emb.tolist()


def _is_procedural(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in PROCEDURAL_KEYWORDS)


# =========================================================
# SEARCH
# =========================================================

def _vector_search(query: str, k: int = 10):
    q_emb = _get_embedding(query, is_query=True)
    count = collection.count()
    if count == 0:
        return {"documents": [[]], "distances": [[]], "metadatas": [[]]}
    return collection.query(
        query_embeddings=[q_emb],
        n_results=min(k, count),
        include=["documents", "distances", "metadatas"],
    )


def _bm25_search(query: str, k: int = 10):
    if _bm25 is None:
        return []
    scores = np.array(_bm25.get_scores(_tokenize(query)))
    if scores.max() > 0:
        scores /= scores.max()
    ranked = np.argsort(scores)[::-1][:k]
    return [(_docs[i], float(scores[i])) for i in ranked]


def _hybrid_search(query: str, alpha: float = 0.8):
    vr = _vector_search(query, k=10)
    br = _bm25_search(query, k=10)
    results: dict[str, dict] = {}

    for doc, dist, meta in zip(
        vr["documents"][0], vr["distances"][0], vr["metadatas"][0]
    ):
        vs = 1 - dist
        results[doc] = {
            "metadata": meta,
            "vector_score": vs,
            "bm25_score": 0,
            "hybrid_score": vs * alpha,
        }

    for doc, bs in br:
        if doc not in results:
            results[doc] = {
                "metadata": _doc_metadata.get(doc, {}),
                "vector_score": 0,
                "bm25_score": bs,
                "hybrid_score": bs * (1 - alpha),
            }
        else:
            results[doc]["bm25_score"] = bs
            results[doc]["hybrid_score"] += bs * (1 - alpha)

    return sorted(results.items(), key=lambda x: x[1]["hybrid_score"], reverse=True)


def _procedural_search(query: str):
    results = _hybrid_search(query)
    proc = [
        (doc, sc)
        for doc, sc in results
        if sc["metadata"].get("step_number", -1) != -1
    ]
    return sorted(proc, key=lambda x: x[1]["metadata"]["step_number"])


def _retrieve(query: str):
    if _is_procedural(query):
        pr = _procedural_search(query)
        if not pr:
            # ── explicit fallback log (visible in server console)
            print("No procedural results found -> fallback to hybrid search")
            return _hybrid_search(query)
        return pr
    return _hybrid_search(query)


# =========================================================
# LLM HELPERS
# =========================================================

def _llm(prompt: str, max_tokens: int = 80) -> str:
    r = ollama.chat(
        model="llama3:instruct",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens},
    )
    return r["message"]["content"].strip()


def _extract_topic(user_input: str) -> str:
    return _llm(
        f"Extract the main topic from this message in 2-4 words maximum.\n"
        f"Return ONLY the topic name. No punctuation, no explanation.\n"
        f"Message: {user_input}\nTopic:",
        max_tokens=20,
    )


def _generate_clarification(user_input: str, topic: str) -> str:
    return _llm(
        f'You are a friendly AI assistant.\nThe user wants help with: "{topic}"\n'
        f'Ask ONE short friendly question about their familiarity with "{topic}".\n'
        f"Do NOT greet them using the topic name.\n"
        f"Return ONLY the question.",
        max_tokens=80,
    )


def _classify_yes_no(text: str) -> str:
    t = text.strip().lower()
    yes_kw = ["yes", "yep", "yeah", "yup", "done", "completed", "finished", "sure",
              "ok", "okay", "alright", "got it", "all done", "complete", "i did",
              "i have", "did it", "works", "worked", "success"]
    no_kw  = ["no", "nope", "nah", "not yet", "not done", "confused", "stuck",
               "struggling", "help", "don't understand", "cannot", "can't",
               "didn't", "doesn't", "failed", "problem", "issue", "error"]
    for kw in yes_kw:
        if re.search(rf"\b{re.escape(kw)}\b", t):
            return "YES"
    for kw in no_kw:
        if re.search(rf"\b{re.escape(kw)}\b", t):
            return "NO"
    raw = _llm(
        f"Classify as YES or NO only. YES = completed/agreed. NO = didn't/needs help.\n"
        f"Reply ONLY with YES or NO.\nReply: \"{text}\"",
        max_tokens=5,
    ).upper()
    return "YES" if "YES" in raw else "NO"


def _classify_proceed_or_new(text: str) -> str:
    t = text.strip().lower()
    proceed_kw = ["proceed", "next", "continue", "yes", "yep", "yeah", "go ahead",
                  "next step", "go on", "keep going", "move on", "forward", "ok",
                  "okay", "sure", "alright", "let's go", "lets go"]
    new_kw = ["new topic", "something else", "different", "no", "nope", "nah", "done",
              "finish", "stop", "exit", "that's all", "thats all", "i'm done", "im done",
              "new question", "another topic"]
    for kw in proceed_kw:
        if re.search(rf"\b{re.escape(kw)}\b", t):
            return "PROCEED"
    for kw in new_kw:
        if re.search(rf"\b{re.escape(kw)}\b", t):
            return "NEW"
    raw = _llm(
        f"Classify as PROCEED or NEW only.\nReply ONLY with PROCEED or NEW.\nReply: \"{text}\"",
        max_tokens=5,
    ).upper()
    return "PROCEED" if "PROCEED" in raw else "NEW"


def _parse_all_steps(query: str, results: list) -> list[str]:
    if not results:
        return []
    context = "\n\n".join(
        [f"Document {i+1}:\n{doc}" for i, (doc, _) in enumerate(results[:6])]
    )
    prompt = (
        f"You are a friendly trainer writing a visual step-by-step guide.\n"
        f"From the context below, extract ALL steps relevant to this task: \"{query}\"\n\n"
        f"Rules:\n"
        f"- Start immediately with: Step 1: (no preamble)\n"
        f"- Each step format:\n\nStep N: <Short action title>\n\n"
        f"<2-4 sentence explanation>\n\n<image URL if present>\n\n"
        f"- Group closely related sub-actions into ONE step (aim 5-8 steps).\n"
        f"- Use ONLY information from context. Do NOT invent.\n"
        f"- Do NOT output 'A Comprehensive Guide', 'Introduction', or preamble.\n\n"
        f"Context:\n{context}\n\nGuide (start with Step 1:):"
    )
    raw = _llm(prompt, max_tokens=1500)
    first = re.search(r"(?i)Step\s+1\s*:", raw)
    if first:
        raw = raw[first.start():]
    parts = re.split(r"(?=Step\s+\d+\s*:)", raw, flags=re.IGNORECASE)
    cleaned = []
    for part in parts:
        part = part.strip()
        if not part or not re.match(r"(?i)^Step\s+\d+\s*:", part):
            continue
        first_line = part.split("\n")[0]
        if any(re.search(p, first_line) for p in GARBAGE_PATTERNS):
            continue
        cleaned.append(part)
    return cleaned


def _generate_focused_answer(query: str, results: list) -> str:
    """
    Non-procedural path: answer ONLY the user's exact question.
    Include image URLs exactly as they appear in the context.
    """
    if not results:
        return "No relevant information found."
    context = "\n\n".join(
        [f"Document {i+1}:\n{doc}" for i, (doc, _) in enumerate(results[:6])]
    )
    prompt = (
        f"You are a friendly and patient trainer.\n"
        f"The user's specific question is: \"{query}\"\n\n"
        f"IMPORTANT RULES:\n"
        f"1. Answer ONLY what the user asked. Do not add unrelated information.\n"
        f"2. Be concise, warm, and direct.\n"
        f"3. If the answer includes image URLs (ending in .jpg/.jpeg/.png/.webp), "
        f"include them on their own line so they can be displayed.\n"
        f"4. Do NOT invent facts — use ONLY the context below.\n\n"
        f"Context:\n{context}\n\nAnswer:"
    )
    return _llm(prompt, max_tokens=1000)


def _generate_challenge_answer(
    user_challenge: str, current_step_raw: str, original_query: str, results: list
) -> str:
    if not results:
        return "I could not find more details in the documents."
    context = "\n\n".join(
        [f"Document {i+1}:\n{doc}" for i, (doc, _) in enumerate(results[:6])]
    )
    prompt = (
        f"You are a friendly and patient trainer helping a user stuck on:\n"
        f"User's challenge: \"{user_challenge}\"\n\n"
        f"This is part of the step:\n\"{current_step_raw}\"\n\n"
        f"Give a thorough detailed explanation focused ONLY on the user's challenge.\n"
        f"Break down into numbered sub-steps if needed. Explain WHY each action matters.\n"
        f"Include image URLs exactly as they appear — one per line.\n\n"
        f"Context:\n{context}\n\nDetailed explanation:"
    )
    return _llm(prompt, max_tokens=1500)


# =========================================================
# SESSION HELPERS
# =========================================================

def _new_session() -> dict:
    return {
        "awaiting_confirmation": False,
        "awaiting_goal": False,
        "awaiting_step_confirmation": False,
        "awaiting_challenge": False,
        "awaiting_proceed_or_new": False,
        "pending_topic": None,
        "last_rag_results": [],
        "last_query": "",
        "current_step_index": 0,
        "all_steps": [],
        "total_steps": 0,
    }


def _get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = _new_session()
    return sessions[session_id]


def _full_reset(s: dict):
    s.update(_new_session())


# =========================================================
# PYDANTIC MODELS
# =========================================================

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    state: str
    images: list[str] = []   # top-level image URLs for non-step answers


class SessionResponse(BaseModel):
    session_id: str


# =========================================================
# ROUTES
# =========================================================

@app.post("/session", response_model=SessionResponse)
def create_session():
    sid = str(uuid.uuid4())
    sessions[sid] = _new_session()
    return {"session_id": sid}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    s = _get_session(req.session_id)
    user_input = req.message.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    # ── STEP A: familiarity reply ────────────────────────────────────────
    if s["awaiting_confirmation"]:
        topic = s["pending_topic"]
        t = user_input.lower()

        experienced_kw = ["yes", "yep", "yeah", "yup", "i have", "familiar", "experienced",
                          "i know", "i use", "i used", "worked with", "know it", "comfortable",
                          "a lot", "many times", "before"]
        new_kw = ["no", "nope", "nah", "never", "new", "beginner", "first time", "don't know",
                  "no idea", "not familiar", "not used", "haven't", "just started", "starting", "learning"]
        vague_kw = ["ok", "okay", "alright", "sure", "fine", "maybe", "i think", "not sure",
                    "hmm", "um", "hi", "hello", "hey", "thanks"]

        decision = None
        for kw in experienced_kw:
            if re.search(rf"\b{re.escape(kw)}\b", t):
                decision = "PROCEED"; break
        if not decision:
            for kw in new_kw:
                if re.search(rf"\b{re.escape(kw)}\b", t):
                    decision = "EXPLAIN_FIRST"; break
        if not decision:
            for kw in vague_kw:
                if re.search(rf"\b{re.escape(kw)}\b", t):
                    decision = "CLARIFY"; break
        if not decision:
            raw = _llm(
                f"User topic: {topic}\nUser reply: {user_input}\n"
                f"Classify into ONE word: PROCEED, EXPLAIN_FIRST, or CLARIFY\n"
                f"Return ONLY one word.",
                max_tokens=5,
            ).upper()
            decision = "PROCEED" if "PROCEED" in raw else ("EXPLAIN_FIRST" if "EXPLAIN" in raw else "CLARIFY")

        if decision in ("PROCEED", "EXPLAIN_FIRST"):
            topic_lower = topic.lower()
            examples_list = next(
                (exs for key, exs in TOPIC_EXAMPLES.items() if key in topic_lower),
                [f"Setting up {topic}", f"Using {topic} step by step", f"Troubleshooting {topic}"]
            )
            bullets = "\n".join(f"• {e}" for e in examples_list)
            ack = (
                f"No worries — it's great that you're getting started with {topic}!"
                if decision == "EXPLAIN_FIRST"
                else f"Great, glad you have some experience with {topic}!"
            )
            reply = (
                f"{ack}\n\nHere are a few things people commonly need help with for {topic}:\n\n"
                f"{bullets}\n\nWhat specific task do you need help with today?"
            )
            s["awaiting_confirmation"] = False
            s["awaiting_goal"] = True
            return {"reply": reply, "state": "awaiting_goal", "images": []}

        else:
            reply = _llm(
                f'The user said: "{user_input}" when asked if familiar with {topic}.\n'
                f'Reply naturally in one short friendly sentence, then gently re-ask:\n'
                f'"Have you worked with {topic} before, or is this new to you?"\n'
                f"Keep it warm and under 40 words.",
                max_tokens=60,
            )
            return {"reply": reply, "state": "awaiting_confirmation", "images": []}

    # ── STEP B: user states goal ─────────────────────────────────────────
    if s["awaiting_goal"]:
        topic = s["pending_topic"]
        vague = _llm(
            f'The user was asked what they want to do with "{topic}".\n'
            f'They replied: "{user_input}"\n'
            f"Is this a real task/goal or just an acknowledgment?\n"
            f"Answer ONE word only: TASK or VAGUE",
            max_tokens=5,
        ).upper()

        if "VAGUE" in vague:
            reply = _llm(
                f'The user replied "{user_input}" when asked what to do with {topic}.\n'
                f"Gently re-ask what specific task they need help with.\n"
                f"Give examples. Keep it friendly and under 50 words.",
                max_tokens=80,
            )
            return {"reply": reply, "state": "awaiting_goal", "images": []}

        full_query = f"{topic} {user_input}"
        results = _retrieve(full_query)[:10]
        s["last_rag_results"] = [(doc, sc) for doc, sc in results]
        s["last_query"] = full_query
        s["awaiting_goal"] = False

        if _is_procedural(user_input):
            all_steps = _parse_all_steps(full_query, results)
            if not all_steps:
                _full_reset(s)
                return {
                    "reply": "I could not find any steps for that in the documents. Please try rephrasing.",
                    "state": "idle",
                    "images": [],
                }
            s["all_steps"] = all_steps
            s["total_steps"] = len(all_steps)
            s["current_step_index"] = 0
            total = len(all_steps)
            intro = f"Great! I found **{total} step{'s' if total > 1 else ''}** for this task. Let's go one at a time. 🚀\n\n"
            first_step = _format_step(all_steps[0], 0, total)
            reply = intro + first_step + _ask_completion()
            # Extract images from first step
            _, imgs = extract_images_with_context(all_steps[0])
            image_urls = [i["url"] for i in imgs]
            s["awaiting_step_confirmation"] = True
            return {"reply": reply, "state": "awaiting_step_confirmation", "images": image_urls}
        else:
            reply = _generate_focused_answer(full_query, results)
            _, imgs = extract_images_with_context(reply)
            image_urls = [i["url"] for i in imgs]
            # Strip URLs from reply text since they're in images field
            clean_reply, _ = extract_images_with_context(reply)
            _full_reset(s)
            return {"reply": clean_reply, "state": "idle", "images": image_urls}

    # ── STEP C: did user complete the step? ──────────────────────────────
    if s["awaiting_step_confirmation"]:
        decision = _classify_yes_no(user_input)
        idx = s["current_step_index"]
        total = s["total_steps"]

        if decision == "YES":
            if idx + 1 < total:
                reply = _ask_next_step(idx, total)
                s["awaiting_step_confirmation"] = False
                s["awaiting_proceed_or_new"] = True
                return {"reply": reply, "state": "awaiting_proceed_or_new", "images": []}
            else:
                _full_reset(s)
                return {
                    "reply": "🎉 **Outstanding! You've completed ALL steps!**\n\nYou did a fantastic job! Whenever you're ready, just type a new question.",
                    "state": "idle",
                    "images": [],
                }
        else:
            s["awaiting_step_confirmation"] = False
            s["awaiting_challenge"] = True
            return {
                "reply": "No worries at all! 🙌 That's what I'm here for.\n\nWhich part or concept did you find challenging?\n\nDescribe it and I'll help you with exactly that part.",
                "state": "awaiting_challenge",
                "images": [],
            }

    # ── STEP D: focused help for challenge ───────────────────────────────
    if s["awaiting_challenge"]:
        idx = s["current_step_index"]
        steps = s["all_steps"]
        total = s["total_steps"]
        current_step_raw = steps[idx]
        original_query = s["last_query"] or s["pending_topic"] or ""
        challenge_query = f"{original_query} {current_step_raw} {user_input}".strip()
        results = _retrieve(challenge_query)[:10]
        s["last_rag_results"] = results
        detailed = _generate_challenge_answer(user_input, current_step_raw, original_query, results)
        step_display = _format_step(current_step_raw, idx, total)
        reply = (
            "Got it! Here's a detailed explanation to help you through this part 💡\n\n---\n\n"
            + detailed
            + "\n\n---\n\n**Here is the step again so you can try once more:**\n\n"
            + step_display
            + _ask_completion()
        )
        # Collect images from both detailed answer and current step
        _, imgs1 = extract_images_with_context(detailed)
        _, imgs2 = extract_images_with_context(current_step_raw)
        seen: set[str] = set()
        image_urls = []
        for img in imgs1 + imgs2:
            if img["url"] not in seen:
                seen.add(img["url"])
                image_urls.append(img["url"])

        s["awaiting_challenge"] = False
        s["awaiting_step_confirmation"] = True
        return {"reply": reply, "state": "awaiting_step_confirmation", "images": image_urls}

    # ── STEP E: proceed or new topic ─────────────────────────────────────
    if s["awaiting_proceed_or_new"]:
        decision = _classify_proceed_or_new(user_input)
        if decision == "PROCEED":
            s["current_step_index"] += 1
            idx = s["current_step_index"]
            total = s["total_steps"]
            steps = s["all_steps"]
            if idx < total:
                _, imgs = extract_images_with_context(steps[idx])
                image_urls = [i["url"] for i in imgs]
                reply = _format_step(steps[idx], idx, total) + _ask_completion()
                s["awaiting_proceed_or_new"] = False
                s["awaiting_step_confirmation"] = True
                return {"reply": reply, "state": "awaiting_step_confirmation", "images": image_urls}
            else:
                _full_reset(s)
                return {
                    "reply": "🎉 **You've completed all the steps!**\n\nGreat work! Feel free to ask a new question anytime.",
                    "state": "idle",
                    "images": [],
                }
        else:
            _full_reset(s)
            return {
                "reply": "Wonderful! 🌟 You've done a great job today!\n\nWhenever you're ready, just type your next question or pick a topic below.",
                "state": "idle",
                "images": [],
            }

    # ── STEP F: new topic / first message ────────────────────────────────
    if _is_casual(user_input):
        t = user_input.strip().lower()
        if re.match(r"^\s*(hi|hello|hey|hiya|howdy|greetings|good\s+(morning|afternoon|evening|day))\s*[!.,]?\s*$", t):
            reply = "Hello! 👋 I'm your AI training assistant. What topic would you like help with today?"
        elif re.match(r"^\s*(bye|goodbye|see you|cya|later|take care)\s*[!.,]?\s*$", t):
            reply = "Goodbye! 👋 Feel free to come back anytime you need help."
        elif re.match(r"^\s*(how are you|how r u|how do you do|what's up|wassup|sup)\s*[?!.,]?\s*$", t):
            reply = "I'm doing great, thanks for asking! 😊 What topic would you like help with today?"
        else:
            reply = "Got it! 😊 What topic would you like help with today?"
        return {"reply": reply, "state": "idle", "images": []}

    topic = _extract_topic(user_input)
    s["pending_topic"] = topic
    question = _generate_clarification(user_input, topic)
    s["awaiting_confirmation"] = True
    return {"reply": question, "state": "awaiting_confirmation", "images": []}


# =========================================================
# FORMAT HELPERS
# =========================================================

def _format_step(step_text: str, idx: int, total: int) -> str:
    filled = round((idx + 1) / total * 10)
    bar = "🟩" * filled + "⬜" * (10 - filled)
    return (
        f"### 📍 Step {idx + 1} of {total}\n"
        f"{bar} `{idx + 1}/{total}`\n\n---\n\n"
        + step_text
    )


def _ask_completion() -> str:
    return (
        "\n\n---\n\n"
        "✅ **Have you completed this step?**\n\n"
        "Reply **yes** when done, or **no** if you need help with this part."
    )


def _ask_next_step(current: int, total: int) -> str:
    remaining = total - current - 1
    if remaining == 1:
        msg = "There is **1 step remaining**."
    elif remaining > 1:
        msg = f"There are **{remaining} steps remaining**."
    else:
        msg = "This is the **last step**!"
    return (
        f"Great job! 🎉 {msg}\n\n"
        "- Reply **next** to continue\n"
        "- Reply **new topic** to start something different"
    )


# =========================================================
# DOCUMENT UPLOAD  (multi file type, delegates to ingest.py)
# =========================================================

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (.txt, .pdf, .docx, .md, .csv),
    chunk it, embed it, and store in ChromaDB.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. "
                   f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    content = await file.read()

    # Write to a temp file so ingest.py can use its path-based extractors
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest_file(tmp_path)
        # Rename the "source" to the real filename
        result["source"] = file.filename
    finally:
        os.unlink(tmp_path)

    _rebuild_bm25()
    return {
        "message": f"Uploaded {result['chunks_added']} chunks from '{file.filename}'",
        "chunks_added": result["chunks_added"],
        "source": file.filename,
    }


@app.get("/indexed-sources")
def indexed_sources():
    """List all filenames that have been indexed."""
    data = collection.get(include=["metadatas"])
    sources = sorted({m.get("source", "unknown") for m in data.get("metadatas", [])})
    return {"sources": sources, "total_docs": collection.count()}


@app.get("/health")
def health():
    return {"status": "ok", "docs_indexed": len(_docs)}
