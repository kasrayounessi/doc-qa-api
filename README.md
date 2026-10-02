# doc-qa-api

A production-quality document question-answering API. Upload a document (PDF or JSON) and a list of questions; the service answers each question using only evidence retrieved from that document.

---

## Architecture

```
PDF ─────► PDF Loader ──────┐
                            │
JSON ────► JSON Loader ─────┤
                            ▼
                   Normalized Documents
                   (InternalDocument)
                            │
                            ▼
                RecursiveCharacterTextSplitter
                            │
                            ▼
                     OpenAI Embeddings
                  (text-embedding-3-small)
                            │
                            ▼
                    FAISS (in-memory)
                            │
           Questions ───────┤
                            ▼
                    DocumentRetriever
                     (top_k chunks)
                            │
                            ▼
                    build_context()
                    (labeled chunks)
                            │
                            ▼
                  Grounded Prompt
             (system: answer ONLY from context)
                            │
                            ▼
                      GPT-4o-mini
                     (temperature=0)
                            │
                            ▼
               Answer + Source References
```

The document is parsed, chunked, and embedded **once per request**. All questions reuse the same FAISS index.

---

## Setup

### Prerequisites

- Python 3.11+ and an OpenAI API key _(API-only path)_
- Docker and Docker Compose _(full-stack path)_

---

### Option A — API only (Python, no Docker)

The project requires Python 3.11+. macOS ships with Python 3.9; use the
Homebrew-installed Python 3.12 (or later) explicitly.

```bash
git clone https://github.com/kasrayounessi/doc-qa-api.git
cd doc-qa-api

# Use the Homebrew Python — NOT the macOS system Python
/opt/homebrew/bin/python3.12 -m venv .venv

# Install all dependencies directly through the venv pip
.venv/bin/pip install -e ".[dev]"

cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

> **Why not `python -m venv`?** macOS system Python is 3.9 and will be rejected
> by the `requires-python = ">=3.11"` constraint. Always use the full path to
> the Homebrew binary, or activate the venv before calling pip:
> `source .venv/bin/activate && pip install -e ".[dev]"`

---

### Option B — Full stack: API + React client (requires Docker)

```bash
git clone https://github.com/kasrayounessi/doc-qa-api.git
cd doc-qa-api

cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

docker compose up --build
```

- API: `http://localhost:8000`
- Client: `http://localhost:3000`

Both services build and start with the single command above. The API image compiles `faiss-cpu` natively (gcc/g++ installed inside the container); expect a 2–3 minute first build.

---

### Local client development (no Docker)

If you want to iterate on the React UI without rebuilding Docker images:

```bash
# Terminal 1 — start the API
uvicorn app.main:app --reload

# Terminal 2 — start the Vite dev server
cd client && npm install && npm run dev
```

The Vite dev server proxies all `/v1` requests to `http://localhost:8000`, so no CORS configuration is needed. Visit `http://localhost:5173`.

---

## React Client

The client lives in `client/` and is a plain Vite + React app with no UI framework dependencies.

**UI flow:**
1. Select a **document** (PDF or JSON) — or click one of the sample buttons to load the bundled fixtures.
2. Select a **questions file** (JSON array) — or click a sample.
3. Click **Ask**. A loading spinner appears while the API processes the request.
4. Answers render as cards: question, grounded answer, and source chips (page number for PDFs, JSON path for JSON documents).

The bundled sample documents are `pdf_sample.pdf` and `json_sample.json` (served from `client/public/samples/`). Clicking a sample button fetches the file client-side and feeds it through the same upload path as a manually chosen file.

---

## Environment Variables

| Variable             | Default                  | Description                                  |
| -------------------- | ------------------------ | -------------------------------------------- |
| `OPENAI_API_KEY`     | _(required)_             | OpenAI API key for embeddings and generation |
| `GENERATION_MODEL`   | `gpt-4o-mini`            | LLM used for answer generation               |
| `EMBEDDING_MODEL`    | `text-embedding-3-small` | Embedding model for vector indexing          |
| `CHUNK_SIZE`         | `1000`                   | Maximum characters per text chunk            |
| `CHUNK_OVERLAP`      | `150`                    | Character overlap between adjacent chunks    |
| `RETRIEVAL_TOP_K`    | `4`                      | Number of chunks retrieved per question      |
| `MAX_UPLOAD_SIZE_MB` | `50`                     | Maximum document upload size in megabytes    |
| `LOG_LEVEL`          | `INFO`                   | Python logging level                         |

---

## Running the API

```bash
uvicorn app.main:app --reload
```

The service starts at `http://localhost:8000`.

---

## Example Request

```bash
curl -X POST http://localhost:8000/v1/qa \
  -F "document_file=@client/public/samples/pdf_sample.pdf" \
  -F "questions_file=@client/public/samples/pdf_questions.json"
```

**questions.json** — Format A (with IDs):

```json
[
  { "id": "q1", "question": "What is the company password policy?" },
  { "id": "q2", "question": "Is MFA required?" }
]
```

**questions.json** — Format B (strings only):

```json
["What is the company password policy?", "Is MFA required?"]
```

Both formats are accepted and normalized internally.

---

## Example Response

```json
{
  "results": [
    {
      "id": "q1",
      "question": "What is the company password policy?",
      "answer": "Passwords must contain at least 12 characters, including uppercase, lowercase, digits, and special characters.",
      "sources": [
        { "source": "security_policy.pdf", "page": 4, "json_path": null }
      ]
    },
    {
      "id": "q2",
      "question": "Is MFA required?",
      "answer": "The document does not contain sufficient information to answer this question.",
      "sources": []
    }
  ]
}
```

Source metadata is populated from the chunks actually retrieved and passed to the model — page numbers come from PDF metadata, `json_path` from JSON section keys.

---

## Running Tests

```bash
pytest
```

All tests mock OpenAI — no API key required and no API cost is incurred.

```bash
pytest -v          # verbose output
pytest -x          # stop on first failure
```

---

## Evaluation

A lightweight evaluation suite is included to verify retrieval quality and grounding behavior.

```bash
# Deterministic only — no API cost
python -m scripts.evaluate

# With live LLM calls (requires OPENAI_API_KEY, incurs cost)
python -m scripts.evaluate --live
```

Expected output:

```
============================================================
Retrieval Evaluation (deterministic, no API cost)
============================================================
  HIT:  [Direct factual retrieval] 'Is MFA required for administrators?'
  HIT:  [Semantic retrieval (different wording)] 'Who leads the organization?'
  HIT:  [Semantic retrieval (paraphrase of 'founded')] 'When was the organization established?'

Retrieval hit rate: 3/3

============================================================
QA Grounding Evaluation (mocked LLM, no API cost)
============================================================
  Supported questions:
  PASS: [supported_1] 'Is MFA required for administrators?'
  PASS: [supported_2] 'What authentication provider is used?'
  Unsupported questions (expected abstention):
  PASS: [unsupported_1] 'What is the CEO's annual salary?'

Supported cases correctly answered: 2/2
Unsupported cases correctly declined: 1/1

All deterministic checks passed.
```

The deterministic retrieval section uses hash-seeded embeddings and tests that signal chunks outrank noise chunks for semantically related queries. The grounding section verifies that context flows correctly through the pipeline and that unsupported questions produce abstention rather than fabricated answers.

---

## Docker

**Full stack (API + React client) — recommended:**

```bash
docker compose up --build
```

**API only:**

```bash
docker build -t document-qa .

docker run \
  -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  document-qa
```

---

## Design Decisions

### Why FAISS?

Each request processes one document. An in-process vector index avoids external infrastructure entirely, matches the stateless HTTP semantics of the endpoint, and can be replaced by changing a single function (`build_vector_store`) if persistent or multi-tenant storage becomes necessary.

### Why explicit RAG stages rather than an opaque chain?

```python
retrieved_chunks = retriever.retrieve(question)
context         = build_context(retrieved_chunks)
answer          = qa_service.generate(question, context)
```

Each stage is independently testable. Retrieval failures (wrong chunks) are diagnosable separately from generation failures (poor answers given good context). `RetrievalQA` and similar abstractions fuse these stages into a black box that makes evaluation and debugging harder.

### Why no agents or LangGraph?

This is a deterministic retrieve-and-answer workflow. There is no branching, tool selection, or multi-step planning. Adding an agent would increase latency, cost, and complexity without addressing any requirement.

### Why source metadata?

Document QA answers should be traceable to specific pages or sections so they can be verified by a human. Source references also make it straightforward to detect when the model is drawing from the context versus hallucinating.

### Why JSON flattening rather than `str(json_data)`?

A nested JSON object converted to a raw Python string loses structural context — `True` becomes the Python literal rather than a semantic fact, and key hierarchy collapses into an unreadable blob. The recursive flattener preserves paths like `security.authentication.mfa_required: true` so both the embedding and the LLM receive semantically coherent text. Keys are grouped by top-level section before chunking to avoid splitting logically related facts across chunk boundaries.

### Why temperature 0.0?

The grounding requirement is strict: answer only from the provided context. Higher temperatures increase the probability of the model drawing on parametric knowledge instead of the supplied evidence.

---

## Limitations

- **Scanned / image-only PDFs**: `pypdf` extracts embedded text only. PDFs that consist entirely of scanned images require an OCR preprocessing step (e.g. Tesseract + `pdf2image`) before ingestion. The service returns a clear 400 error for such files rather than silently returning empty answers.
- **Complex PDF layouts**: Tables, multi-column layouts, and heavily formatted PDFs may extract poorly with `pypdf`. A layout-aware parser such as `pdfplumber` or `unstructured` would improve extraction fidelity.
- **One LLM call per question**: Each question in the batch makes an independent generation call. For large question sets, batching or parallelizing these calls would reduce latency.
- **Ephemeral vector index**: The FAISS index is built in memory and discarded after the request. The same document uploaded twice is embedded twice. For repeated queries over the same document, caching the index would reduce cost.
- **Chunk parameters are tuned to defaults**: `CHUNK_SIZE=1000` and `CHUNK_OVERLAP=150` are reasonable starting points. Optimal values depend on the typical document length and question type; evaluation against a representative dataset would guide tuning.

---

## Production Evolution

At larger scale these additions would be appropriate — none are needed for the current workload:

- **Persistent vector storage** — pgvector, Qdrant, or Pinecone for multi-user or multi-document retrieval
- **Async ingestion** — background workers for large documents so the HTTP response is not blocked
- **OCR / layout-aware parsing** — Tesseract or `unstructured` for scanned PDFs and complex layouts
- **Hybrid retrieval + reranking** — BM25 + dense retrieval with a cross-encoder reranker for higher precision
- **Document-level cache** — cache chunk embeddings keyed by document hash to avoid re-embedding identical uploads
- **Observability** — structured traces (OpenTelemetry / LangSmith) for per-request latency and retrieval quality monitoring
- **Offline evaluation** — a labeled benchmark dataset and automated regression testing on each model or parameter change
