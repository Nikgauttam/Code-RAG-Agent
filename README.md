# CodeRAG — Context-Aware AI Code Agent

[![CI](https://github.com/Nikgauttam/Code-RAG-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Nikgauttam/Code-RAG-Agent/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![types](https://img.shields.io/badge/typed-mypy%20strict-brightgreen)

A local, privacy-first RAG pipeline that answers natural-language questions about any Python codebase. Combines **semantic retrieval** (sentence-transformer embeddings + FAISS), a **structural dependency graph** (AST imports + caller→callee call edges), and a **cross-encoder reranker** before grounding the answer in a local LLM (Ollama). Zero API cost. Data never leaves your machine.

## Demo

```
$ python cli.py --repo /path/to/flask

Indexing codebase...
Index built: 1624 chunks across 83 files.

>>> how does Flask handle sessions?

╭─────────────────────────── Answer ────────────────────────────╮
│ Flask handles sessions via the SecureCookieSessionInterface   │
│ (sessions.py L284). Sessions are stored as signed cookies     │
│ using itsdangerous. The open_session() method deserializes    │
│ the cookie on each request (L330-L352), and save_session()    │
│ re-signs and sets the cookie in the response (L354-L395).     │
│ The signing key is the app's SECRET_KEY config value.         │
╰───────────────────────────────────────────────────────────────╯
Sources:
  • src/flask/sessions.py
  • src/flask/ctx.py
  • src/flask/app.py

>>> explain SecureCookieSessionInterface
>>> find usages of url_for
```

## Benchmark Results

Evaluated on [pallets/flask](https://github.com/pallets/flask) — 30 hand-labeled questions, 1,624 chunks across 83 files:

| Strategy | Recall@5 | Recall@10 | MRR | p95 ms |
|----------|:--------:|:---------:|:---:|:------:|
| BM25 (baseline) | 0.500 | 0.633 | 0.252 | <1 |
| Semantic only | 0.633 | 0.667 | 0.437 | 19 |
| Hybrid | 0.633 | 0.667 | 0.437 | 6 |
| **Hybrid + Rerank** ✅ | **0.667** | **0.667** | **0.536** | 143 |

> Hybrid+rerank achieves **+112% MRR over BM25** and **+23% over semantic-only**.

Reproduce: `python -m evaluation.run_eval --repo /tmp/flask_eval --eval flask`

The goal: don't just find lexically similar code — find code that is *structurally* relevant, the way a senior engineer would.

## Features

- **Hybrid retrieval** — semantic vector search (FAISS) + AST call/import graph traversal, weighted 70/30
- **Cross-encoder reranking** — ms-marco reranker re-orders top-20 candidates for precision (+23% MRR over semantic-only)
- **AST-based parsing** — extracts functions, classes, methods, imports, and caller→callee edges with fully qualified names
- **Incremental indexing** — SHA-256 content hashing means second run loads from disk instantly (no re-embedding)
- **Grounded generation** — LLM can only answer from retrieved code chunks, cannot hallucinate functions that don't exist
- **Fully local** — Llama3 via Ollama, no API key, no internet required, your code never leaves your machine
- **Production API** — FastAPI with `/healthz`, `/readyz`, `/ask`, `/ask_stream` (Server-Sent Events)
- **Interactive CLI** — Rich-formatted REPL with explain / trace / find-usages commands
- **Docker Compose** — one command brings up Ollama + API together
- **CI/CD** — ruff + mypy + pytest matrix across Python 3.10 / 3.11 / 3.12

---

## Architecture

```
                                              ┌────────────────────┐
                                              │   User query       │
                                              └─────────┬──────────┘
                                                        │
        ┌───────────────────────────────────────────────┼───────────────────────────────────────────┐
        │ Indexing pipeline (cold + warm/incremental)   │                                           │
        │                                               ▼                                           │
        │  Repo files                            ┌──────────────┐                                   │
        │   │ AST visitor  (FQN, scope-aware)    │  Embed query │                                   │
        │   ▼                                    └──────┬───────┘                                   │
        │  Symbol table (typed)                          │                                          │
        │   ├──► FileGraph    (resolved imports)          ▼                                         │
        │   └──► SymbolGraph  (caller → callees)  ┌──────────────┐    ┌──────────────────────┐      │
        │                                          │ FAISS Top-K  │ ─► │ FileGraph DFS expand │      │
        │  Chunker (fn / class / method)           └──────┬───────┘    └──────────┬───────────┘      │
        │   │ stable_id, content_hash                     ▼                       ▼                  │
        │   ▼                                      ┌─────────────────────────────────┐               │
        │  Embedder (L2-normalised, hash-cached)   │ Hybrid score (0.7·sem + 0.3·gr) │               │
        │   ▼                                      └─────────────────┬───────────────┘               │
        │  IndexIDMap2(IndexFlatIP)                                 ▼                                │
        │  + JSON manifest + .npz embeddings        ┌─────────────────────────────────┐              │
        │                                           │ Cross-encoder rerank (MS-MARCO) │              │
        │                                           └─────────────────┬───────────────┘              │
        │                                                             ▼                              │
        │                                           ┌─────────────────────────────────┐              │
        │                                           │ Grounded prompt → Ollama LLM    │              │
        │                                           │  (sync / async / streaming)     │              │
        │                                           └─────────────────┬───────────────┘              │
        └─────────────────────────────────────────────────────────────┴──────────────────────────────┘
                                                                      ▼
                                                              { answer, sources }
```

## Why this design

| Choice | Reason |
|---|---|
| AST visitor with scope stack | Captures `caller → callee` edges, not just "this file calls X somewhere" |
| Separate `FileGraph` + `SymbolGraph` | Mixing file-paths and symbol-names in one dict is a typing/correctness bug |
| Module-path import resolution | Avoids basename collisions (`pkg/utils.py` vs `other/utils.py`) |
| Per-method chunking + class header | A 1000-line class shouldn't be one embedding |
| Stable chunk IDs (hash → int64) | Survives serialization; lets FAISS delete-by-id work; powers incremental reindex |
| Content-hash incremental reindex | Warm reindex re-embeds only the files that changed (typically 1–2, not all) |
| Hybrid score (semantic + graph) | Best of both: lexical similarity *and* call-site proximity |
| Cross-encoder rerank | Bi-encoder is fast but coarse; the cross-encoder fixes top-K ordering |
| `IndexIDMap2(IndexFlatIP)` | Cosine similarity (vectors are L2-normalised) + per-id delete support |
| JSON manifest + `.npz` (no pickle) | Pickle = arbitrary code execution on load; JSON+npz is auditable + safe |
| Local Ollama LLM | Zero per-query cost, no data leaves the machine |

---

## Quickstart

### Local

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# (optional) dev tools — ruff, mypy, pytest, rank_bm25:
pip install -e ".[dev]"

# 2. Run a local LLM (separate terminal)
ollama pull llama3
ollama serve

# 3. Index a repo + start interactive CLI
python cli.py --repo /path/to/your/repo

# 4. Or run as an HTTP API
CODE_AGENT_REPO=/path/to/your/repo uvicorn api:app --reload
```

### Docker (one command, brings up Ollama + API)

```bash
REPO_PATH=/path/to/your/repo docker compose up --build
# then in another terminal:
docker exec -it code-agent-ollama ollama pull llama3
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"query":"how does the embedder work?"}' | jq
```

### CLI commands

```
>>> how does the embedder work?            # free-form RAG question
>>> explain CodeChunker                    # structured symbol explanation
>>> trace generate_answer                  # symbol-graph trace
>>> find usages of VectorStore             # JSON usage report (symbol-table backed)
>>> help                                   # show available commands
>>> exit
```

### HTTP API

```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "where is the FAISS index built?"}'

# Streaming (Server-Sent Events)
curl -N -X POST localhost:8000/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"query": "explain the indexing pipeline"}'

# Liveness / readiness
curl localhost:8000/healthz
curl localhost:8000/readyz
```

### Configuration (env vars)

All knobs are in `core/config.py`. Override any of:

| Variable | Default |
|---|---|
| `CODE_AGENT_REPO` | *required* |
| `CODE_AGENT_MODEL` | `llama3` |
| `CODE_AGENT_LLM_URL` | `http://localhost:11434` |
| `CODE_AGENT_TOP_K` | `5` |
| `CODE_AGENT_RERANK_POOL` | `20` |
| `CODE_AGENT_FINAL_K` | `8` |
| `CODE_AGENT_SEMANTIC_WEIGHT` | `0.7` |
| `CODE_AGENT_GRAPH_WEIGHT` | `0.3` |
| `CODE_AGENT_GRAPH_MAX_DEPTH` | `2` |
| `CODE_AGENT_EMBED_MODEL` | `all-MiniLM-L6-v2` |
| `CODE_AGENT_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `CODE_AGENT_STORAGE` | `storage` |
| `CODE_AGENT_LLM_TIMEOUT` | `60.0` |
| `CODE_AGENT_LLM_RETRIES` | `2` |
| `CODE_AGENT_LLM_TEMPERATURE` | `0.2` |
| `CODE_AGENT_CORS` | `*` |
| `CODE_AGENT_LOG` | `WARNING` |

---

## Project layout

```
core/
  config.py          Central typed config + env-var loader
  parser/            AST visitor → typed FileSymbols (defs, methods, calls, imports)
  graph/             FileGraph (resolved imports) + SymbolGraph (caller→callees)
  retrieval/         Chunker (stable IDs, hashes), embedder, FAISS vector store (IDMap2)
  rerank/            Cross-encoder reranker
  llm/               httpx-based Ollama client (sync / async / streaming, retries)
  pipeline/          End-to-end indexing + retrieval orchestration
agent/               High-level CodeAgent facade with command registry
api.py               FastAPI app (lifespan, /healthz, /readyz, /ask, /ask_stream)
cli.py               Rich-formatted interactive REPL
main.py              CLI entrypoint
storage/             JSON manifest + .npz embeddings + FAISS index (gitignored)
tests/               pytest suite (parser, graph, chunker, vector store, agent, pipeline)
evaluation/          Eval harness (BM25, semantic, hybrid, hybrid+rerank)
```

---

## Evaluation

The harness measures **retrieval quality** independently of the LLM, so
results are deterministic and don't require Ollama running.

```bash
python -m evaluation.run_eval                  # eval on this repo (default)
python -m evaluation.run_eval --repo ../other  # eval on another repo
```

Strategies compared:
- **bm25** — lexical baseline (`rank_bm25`)
- **semantic** — bi-encoder over chunks, FAISS top-k
- **hybrid** — semantic + graph DFS expansion (weighted)
- **hybrid+rerank** — production path with cross-encoder re-ordering

Metrics:
- **Recall@k** — fraction of expected files present in the top-k retrieved files
- **MRR** — mean reciprocal rank of the first relevant file
- **p50 / p95 latency** — wall-clock retrieval time per query

> **Note on the bundled eval set.** The 20 questions in
> `evaluation/eval_set.py` target this repo's own files for reproducibility.
> On a corpus this small, semantic-only saturates Recall@5 = 1.0; the value
> of hybrid + rerank shows up on larger noisier corpora. See
> [Roadmap](#roadmap) for the larger eval set in progress.

## Tests

```bash
pip install -e ".[dev]"
pytest --cov=agent --cov=core --cov-report=term-missing
```

The tiny fixture repos in `tests/conftest.py` cover the two-file import
case, the package + relative-import case, and the same-name shadow file
case (which the previous import resolver got wrong).

## Roadmap

- [x] Eval harness: Recall@k / MRR / latency on a held-out question set
- [x] Pytest suite (parser, graph, chunker, vector store, agent, pipeline)
- [x] **Type-safe** dataclass APIs throughout (`mypy --strict`)
- [x] **Real call graph** (caller → callees, scope-aware)
- [x] **Real import resolution** (module-path, relative-import-aware)
- [x] **Incremental reindex** via per-file content hashes + FAISS IDMap2
- [x] Streaming `/ask_stream` endpoint (Server-Sent Events)
- [x] Dockerfile + docker-compose with Ollama
- [x] BM25 baseline in the eval harness
- [x] Flask benchmark — 30 hand-labeled questions, Recall@5=0.667, MRR=0.536
- [ ] Larger eval set on real OSS repos (Django, FastAPI, Requests) — 100+ Q's
- [ ] Tree-sitter parsing for multi-language support (JS / TS / Go / Rust)
- [ ] Replace `IndexFlatIP` with HNSW for >100k chunks
- [ ] LLM provider abstraction (OpenAI / Ollama / vLLM / Groq)
- [ ] VS Code extension calling the API

---

## Author

**Nikhil Gautam** — B.Tech Computer Science, 2025
Built as a portfolio project demonstrating hybrid RAG, AST-based code intelligence, and production ML engineering.

- GitHub: [@Nikgauttam](https://github.com/Nikgauttam)

---

## License

MIT
