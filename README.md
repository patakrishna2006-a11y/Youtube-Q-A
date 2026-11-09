# YouTube Chat Assistant

A **RAG (Retrieval-Augmented Generation)** web application that lets you chat with any YouTube video using its transcript. Built with Flask, LangChain, ChromaDB, NVIDIA Nemotron-3 Ultra, and Ollama embeddings.

---

## Features

- **🎬 Process any YouTube video** – Paste a URL, fetch its transcript, and build a searchable knowledge base
- **💬 Chat with the video** – Ask questions and get answers grounded in the video's actual content
- **📝 One-click summarization** – Get a concise bullet-point summary instantly
- **🌍 Multi-language support** – 20+ languages including English, Hindi, Spanish, French, German, Japanese, Korean, Arabic, and more
- **🎨 Modern glassmorphism UI** – Dark/light mode, 3D tilt effects, animated background, smooth chat animations
- **🔒 Source citations** – Every answer shows the transcript chunks it was derived from
- **⚡ Session persistence** – Your session survives page refreshes (stored in `sessionStorage`)

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  YouTube    │────▶│  Transcript │────▶│  Text       │────▶│  ChromaDB   │
│  Video URL  │     │  Fetcher    │     │  Splitter   │     │  Vector DB  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                        │
                                                                        ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │◀────│   LLM       │◀────│  MultiQuery │◀────│  Retriever  │
│   (Stream)  │     │  (NVIDIA)   │     │  Retriever  │     │  (k=5)      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**Pipeline:**
1. **Transcript Fetch** – `youtube-transcript-api` downloads captions (manual or auto-generated)
2. **Chunking** – `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
3. **Embedding** – `embeddinggemma-2:latest` via Ollama (local, private)
4. **Vector Store** – ChromaDB persistent collection
5. **Retrieval** – `MultiQueryRetriever` generates multiple query variations for better recall
6. **Generation** – NVIDIA Nemotron-3 Ultra (550B params) via `langchain-nvidia-ai-endpoints`

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Flask 3.x |
| **LLM** | NVIDIA Nemotron-3 Ultra (via `langchain-nvidia-ai-endpoints`) |
| **Embeddings** | `embeddinggemma-2:latest` via Ollama (local) |
| **Vector DB** | ChromaDB 0.5.x (persistent, file-based) |
| **Transcript** | `youtube-transcript-api` 1.2.x |
| **Text Splitting** | `langchain-text-splitters` |
| **Retrieval** | `MultiQueryRetriever` (LangChain) |
| **Frontend** | Vanilla JS + CSS (glassmorphism, 3D tilt, Markdown rendering) |
| **Markdown** | `marked.js` + `DOMPurify` (XSS-safe) |

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| **Python** | 3.10+ | [python.org](https://python.org) |
| **Ollama** | Latest | [ollama.com/download](https://ollama.com/download) |
| **NVIDIA API Key** | – | [NVIDIA NGC](https://org.ngc.nvidia.com/setup/api-key) |

---

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd Youtube-Q&A

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Required: NVIDIA API key for Nemotron-3 Ultra
NVIDIA_API_KEY=your_nvidia_api_key_here

# Optional: Ollama host (default: http://localhost:11434)
OLLAMA_HOST=http://localhost:11434

# Optional: Database folder (default: transcript_db)
DB_DIR=transcript_db
```

### 3. Start Ollama & Pull Embedding Model

```bash
# Start Ollama server (run in a separate terminal)
ollama serve

# Pull the embedding model (first run only, ~1-2 GB)
ollama pull embeddinggemma-2:latest
```

> The app will auto-start Ollama and pull the model if missing, but doing it manually first avoids startup delays.

### 4. Run the App

```bash
python main.py
```

Open **http://localhost:5000** in your browser.

---

## Usage

1. **Paste a YouTube URL** in the sidebar (e.g., `https://www.youtube.com/watch?v=X0btK9X0Xnk`)
2. **Select transcript language** (auto-detected from URL, or choose from 20+ options)
3. **Click "Process Video"** – transcript is fetched, chunked, embedded, and indexed
4. **Ask questions** – type in the chat box and press Enter or click Send
5. **Click "📝 Summarize this video"** for an instant 5-7 bullet summary
6. **Click "Clear chat"** to reset the conversation (keeps the video loaded)

---

## Supported Languages

The app accepts both codes (`en`, `hi`, `fr`) and names (`English`, `Hindi`, `French`):

| Code | Language |
|------|----------|
| `en` | English |
| `hi` | Hindi |
| `fr` | French |
| `es` | Spanish |
| `de` | German |
| `pt` | Portuguese |
| `it` | Italian |
| `ja` | Japanese |
| `ko` | Korean |
| `ar` | Arabic |
| `ru` | Russian |
| `zh` | Chinese |
| `bn` | Bengali |
| `ta` | Tamil |
| `te` | Telugu |
| `mr` | Marathi |
| `gu` | Gujarati |
| `ur` | Urdu |
| `tr` | Turkish |
| `nl` | Dutch |

> **Note:** The video must have captions available in the requested language (manual or auto-generated).

---

## Project Structure

```
Youtube Q&A/
├── main.py                 # Flask app, routes, RAG pipeline
├── ollama_setup.py         # Ollama lifecycle management (start, pull, warm-up)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from .env.example)
├── templates/
│   └── index.html          # Single-page frontend (HTML + vanilla JS)
├── static/
│   └── style.css           # Glassmorphism UI, animations, dark/light mode
└── transcript_db/          # ChromaDB persistent storage (auto-created)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend |
| `POST` | `/api/process` | Fetch transcript, build vector index |
| `POST` | `/api/ask` | Query the RAG system |

### `/api/process` Request

```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "language": "en",           // optional, defaults to "en"
  "session_id": "uuid-v4"     // optional, auto-generated
}
```

**Response (200):**
```json
{
  "session_id": "uuid-v4",
  "video_id": "VIDEO_ID",
  "n_chunks": 42
}
```

### `/api/ask` Request

```json
{
  "session_id": "uuid-v4",
  "question": "What is RAG?"
}
```

**Response (200):**
```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "sources": ["chunk 1 text", "chunk 2 text", "..."]
}
```

---

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `NVIDIA_API_KEY` | *required* | API key for Nemotron-3 Ultra |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `DB_DIR` | `transcript_db` | ChromaDB persistence folder |
| `FLASK_DEBUG` | `True` | Enable/disable debug mode |
| `PORT` | `5000` | Server port |

---

## How It Works (Details)

### Transcript Fetching
- Uses `youtube-transcript-api` v1.2.5 (supports `fetch()` API)
- Tries the requested language; falls back to available languages
- Returns error if no transcript exists for any requested language

### Chunking Strategy
- `RecursiveCharacterTextSplitter`: 1000 chars, 200 overlap
- Separators: `["\n\n", "\n", " ", ""]` (paragraphs → lines → words → chars)
- Preserves video_id and language in chunk metadata

### Multi-Query Retrieval
- `MultiQueryRetriever` uses the LLM to generate 3-5 query variations
- Each variation searches the vector store independently
- Results are deduplicated and combined
- Improves recall for ambiguous or complex questions

### Prompt Engineering
The system prompt enforces:
- **Grounded answers only** – no hallucination, no outside knowledge
- **Concise by default** – 1-3 sentences unless user asks for more
- **Bullet summaries** – max 5-7 points when summarization requested
- **Language matching** – reply in the user's question language
- **Explicit "not covered" response** – "The video doesn't cover this."

### Session Management
- In-memory `SESSIONS` dict: `session_id → {retriever, video_id, n_chunks}`
- Persisted in browser `sessionStorage` (survives refresh)
- New video URL → old DB deleted, fresh index built

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Could not retrieve a transcript` | Video has no captions in requested language. Try `hi` (Hindi auto-generated) or check available languages via YouTube UI. |
| `Ollama not installed` | Install from [ollama.com/download](https://ollama.com/download). Ensure `ollama` is in PATH. |
| `Model not found` | Run `ollama pull embeddinggemma-2:latest` manually. |
| `ChromaDB lock error` (Windows) | The app auto-retries and falls back to a subfolder. Close other apps using the DB folder. |
| `Session expired` on refresh | Ensure `use_reloader=False` in `app.run()` (already set). |
| `NVIDIA API error` | Verify `NVIDIA_API_KEY` in `.env` is valid and has Nemotron access. |
| Slow first request | Ollama warms up the embedding model on startup (~10-30s). Subsequent requests are fast. |

---

## Limitations

- **No live streams** – Only videos with captions (manual or auto-generated)
- **Single video per session** – Processing a new URL replaces the previous index
- **In-memory sessions** – Server restart clears all sessions
- **No user auth** – Anyone with the URL can use the app
- **Transcript quality** – Auto-generated captions may have errors

---

## License

MIT License – feel free to use, modify, and distribute.

---

## Acknowledgments

- [LangChain](https://langchain.com/) for the RAG abstractions
- [NVIDIA](https://www.nvidia.com/) for Nemotron-3 Ultra
- [Ollama](https://ollama.com/) for local embedding models
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) for caption fetching