import gc
import os
import shutil
import time
import uuid

import dotenv
from flask import Flask, jsonify, render_template, request
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings

from ollama_setup import ensure_ollama

dotenv.load_dotenv()

app = Flask(__name__)

# In-memory store: session_id -> {"retriever", "chain", "video_id", "n_chunks"}
SESSIONS = {}

PROMPT = PromptTemplate(
    template="""
You answer questions about a YouTube video using only the transcript excerpts below.

Rules:
- Answer only what was asked. Start with the direct answer; no introduction, no restating the question, no closing remarks.
- Match the length to the question: a simple question gets 1-3 sentences. Go longer only if the user asks to explain, compare, list steps, or summarize.
- If a summary is requested, give at most 5-7 short bullet points.
- Use bullets or bold only when the answer has several distinct parts. Do not use headings unless asked.
- Add an example only if the user asks for one.
- Use only the excerpts. Never add outside knowledge or guess.
- If the excerpts don't contain the answer, reply: "The video doesn't cover this."
- Don't mention "context", "excerpts" or "transcript" in your answer.
- Reply in the same language as the question.

Transcript excerpts:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

# ---------- Models (created once, shared by all sessions) ----------
llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
    max_completion_tokens=5000,
)
EMBED_MODEL = "embeddinggemma-2:latest"
ensure_ollama(EMBED_MODEL)  # starts Ollama and pulls the model if needed
embedding = OllamaEmbeddings(model=EMBED_MODEL)
chain = PROMPT | llm | StrOutputParser()


# ---------- Language handling ----------
LANGUAGE_CODES = {
    "english": "en", "hindi": "hi", "french": "fr", "spanish": "es",
    "german": "de", "portuguese": "pt", "italian": "it", "japanese": "ja",
    "korean": "ko", "arabic": "ar", "russian": "ru", "chinese": "zh",
    "bengali": "bn", "tamil": "ta", "telugu": "te", "marathi": "mr",
    "gujarati": "gu", "urdu": "ur", "turkish": "tr", "dutch": "nl",
}


def normalize_language(value):
    """Accept 'hi', 'Hindi', ' HINDI ', 'en-US' and return a language code."""
    text = (value or "en").strip().lower()
    return LANGUAGE_CODES.get(text, text) or "en"


# ---------- Database handling ----------
import chromadb

try:  # location differs between chromadb versions
    from chromadb.api.shared_system_client import SharedSystemClient
except Exception:
    try:
        from chromadb.api.client import SharedSystemClient
    except Exception:
        SharedSystemClient = None

DB_DIR = os.getenv("DB_DIR", "transcript_db")
CLIENT = None
VECTOR_STORE = None


def _release_db():
    """Close every Chroma handle so Windows lets go of the files."""
    global CLIENT, VECTOR_STORE
    SESSIONS.clear()  # old retrievers point at the old DB
    VECTOR_STORE = None
    if CLIENT is not None:
        for step in (
            lambda: CLIENT.delete_collection("sample"),
            lambda: CLIENT._system.stop(),
            lambda: CLIENT.close(),
        ):
            try:
                step()
            except Exception:
                pass
        CLIENT = None
    if SharedSystemClient is not None:
        try:
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
    gc.collect()


def reset_db():
    """Delete the transcript_db folder completely and create a fresh, empty one.

    Returns the folder path the new database should use.
    """
    _release_db()

    for _ in range(20):  # retry for ~10 s: antivirus / OneDrive may hold files briefly
        shutil.rmtree(DB_DIR, ignore_errors=True)
        if not os.path.exists(DB_DIR):
            break
        time.sleep(0.5)
        gc.collect()

    if os.path.exists(DB_DIR):
        # Still locked by another program: keep working in a fresh subfolder.
        path = os.path.join(DB_DIR, uuid.uuid4().hex)
    else:
        path = DB_DIR
    os.makedirs(path, exist_ok=True)
    return path


# ---------- Pipeline ----------
def build_pipeline(session_id: str, url: str, language: str):
    video_id = YoutubeLoader.extract_video_id(url)

    transcript = YouTubeTranscriptApi().fetch(video_id, languages=[language])
    transcript_text = " ".join(snippet.text for snippet in transcript)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.create_documents(
        [transcript_text],
        metadatas=[{"video_id": video_id, "language": language}],
    )

    # New URL -> delete the old database and create a fresh one.
    # (Done after the transcript is fetched, so a bad URL doesn't wipe a working DB.)
    db_path = reset_db()  # deletes the old transcript_db folder, creates a fresh one

    global CLIENT, VECTOR_STORE
    CLIENT = chromadb.PersistentClient(path=db_path)
    VECTOR_STORE = Chroma(
        client=CLIENT,
        collection_name="sample",
        embedding_function=embedding,
    )
    VECTOR_STORE.add_documents(chunks)
    vector_store = VECTOR_STORE

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        llm=llm,
    )

    SESSIONS[session_id] = {
        "retriever": retriever,
        "video_id": video_id,
        "n_chunks": len(chunks),
    }
    return video_id, len(chunks)


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_video():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    language = normalize_language(data.get("language"))
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not url:
        return jsonify(error="Please enter a YouTube link."), 400

    try:
        video_id, n_chunks = build_pipeline(session_id, url, language)
    except Exception as e:
        return jsonify(error=f"Could not process video: {e}"), 500

    return jsonify(session_id=session_id, video_id=video_id, n_chunks=n_chunks)


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    question = (data.get("question") or "").strip()

    session = SESSIONS.get(session_id)
    if session is None:
        return jsonify(error="Session expired. Please process the video again."), 404
    if not question:
        return jsonify(error="Please enter a question."), 400

    try:
        docs = session["retriever"].invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        answer = chain.invoke({"context": context, "question": question})
    except Exception as e:
        return jsonify(error=f"Something went wrong: {e}"), 500

    return jsonify(answer=answer, sources=[d.page_content for d in docs])


if __name__ == "__main__":
    for old_dir in (DB_DIR, "chroma_db"):  # folders from earlier versions
        shutil.rmtree(old_dir, ignore_errors=True)
    print(f"[main.py v3] running {os.path.abspath(__file__)} | database folder: {DB_DIR}")
    # use_reloader=False: the reloader restarts the server on file changes,
    # which wipes the in-memory sessions and causes "Session expired".
    app.run(debug=True, use_reloader=False, port=5000)