# AI Code Agent — Complete Interview Preparation Guide

> Read this before every interview where you mention this project.
> Every answer here is grounded in the actual code. No bluffing.

---

## 1. THE 30-SECOND PITCH (Say This First)

> "I built a local AI assistant that lets developers ask plain-English questions
> about any Python codebase and get grounded answers with exact file and line
> references. The problem it solves: ChatGPT doesn't know your private code,
> and you can't paste 50,000 lines into a chat window. My system indexes the
> entire repo, finds the most relevant code using a hybrid of semantic search
> and AST-based graph traversal, reranks the results with a cross-encoder, then
> feeds them to a local LLM — zero API cost, data never leaves your machine.
> Benchmarked on Flask: Recall@5 = 0.667, MRR = 0.536 vs BM25 baseline of
> 0.500 / 0.252."

---

## 2. WHAT IS THIS PROJECT?

### Purpose
An AI-powered code Q&A system (RAG — Retrieval Augmented Generation) that:
- Indexes any Python repository into a searchable vector + graph index
- Accepts natural language questions about the codebase
- Retrieves the most relevant code chunks using hybrid search
- Generates grounded answers using a local LLM (no hallucination)

### Real Problem It Solves
| Problem | How This Solves It |
|---------|-------------------|
| ChatGPT doesn't know your private repo | Indexes your repo locally |
| Can't fit 50K lines in a context window | Retrieves only the 8 most relevant chunks |
| LLM hallucinates non-existent functions | Grounded: answer only from retrieved code |
| Keyword search misses structural context | Graph traversal finds callers/importers |
| API costs money, sends data to servers | 100% local via Ollama, zero cost |

### Who Would Use This
- A developer joining a new team who wants to understand an unfamiliar codebase
- An engineer debugging a function and needing to trace all callers
- Anyone who wants to ask "how does X work?" without reading 200 files manually

---

## 3. COMPLETE ARCHITECTURE

```
                        ┌─────────────────────────────────┐
                        │         User Question            │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │   Query Embedder (MiniLM-L6-v2)  │
                        │   384-dim vector representation  │
                        └────────┬───────────────┬────────┘
                                 │               │
               ┌─────────────────▼───┐   ┌───────▼──────────────────┐
               │  FAISS Vector Search │   │  AST Call + Import Graph  │
               │  top-20 semantic     │   │  DFS from seed files      │
               │  matches             │   │  (max depth = 2)          │
               └─────────────────┬───┘   └───────┬──────────────────┘
                                 │               │
                        ┌────────▼───────────────▼────────┐
                        │         Hybrid Score             │
                        │   0.7 × semantic_score           │
                        │ + 0.3 × graph_depth_score        │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │    Cross-Encoder Reranker        │
                        │  ms-marco-MiniLM-L-6-v2         │
                        │  top-8 final chunks              │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │     Llama3 via Ollama (local)    │
                        │  grounded answer + file:line     │
                        │  citations                       │
                        └─────────────────────────────────┘
```

### Five Core Modules

| Module | File | What It Does |
|--------|------|-------------|
| Parser | `core/parser/ast_parser.py` | Extracts functions, classes, methods, imports, call sites from Python AST |
| Dependency Graph | `core/graph/dependency_graph.py` | Builds FileGraph (imports) + SymbolGraph (calls) |
| Chunker | `core/retrieval/chunker.py` | Splits code into embeddable chunks with stable IDs |
| Pipeline | `core/pipeline/codebase_pipeline.py` | Orchestrates index, retrieval, hybrid scoring, generation |
| Agent | `agent/code_agent.py` | Routes commands (explain/trace/usages) and freeform RAG queries |

---

## 4. TECHNOLOGY STACK

### Why Each Tool Was Chosen

| Tool | Version | Why This, Not Alternatives |
|------|---------|---------------------------|
| **sentence-transformers** (MiniLM-L6-v2) | 5.2.0 | 384-dim, fast, free, runs on CPU. GPT embeddings cost money per token. |
| **FAISS** (IndexFlatIP) | 1.13.2 | Sub-millisecond similarity search. Pinecone/Weaviate need internet + cost money. |
| **cross-encoder** (ms-marco-MiniLM-L-6-v2) | — | Fine-tuned on passage ranking. Bi-encoders are faster but less accurate for reranking. |
| **Ollama + Llama3** | 4.7 GB | Free, local, offline. GPT-4 costs ~$0.01/query and sends your private code to OpenAI. |
| **Python AST module** | stdlib | Zero dependencies, accurate for Python. Tree-sitter would add complexity (future work). |
| **FastAPI** | 0.128.0 | Async, auto-generates OpenAPI docs, SSE streaming support. |
| **httpx** | 0.28.1 | Async HTTP for Ollama. `requests` has no async support. |
| **structlog** | 24.4.0 | Structured JSON logging with per-request UUIDs. Better than stdlib logging for production. |
| **rich** | 13.9.4 | Beautiful terminal output, Markdown rendering, syntax highlighting. |

---

## 5. HOW IT WAS BUILT (Step by Step)

### Phase 1 — Parser
- Used Python's built-in `ast.NodeVisitor` pattern
- Built `_SymbolVisitor` with an explicit **scope stack** (`_scope: list[tuple[str, str]]`)
- Tracks `caller_qualname` for every function call — e.g., `Calculator.square` calls `self.multiply`
- Extracts: `SymbolDef` (functions, methods, classes) and `CallSite` (who calls what)
- Handles: async functions, decorators, relative imports (`from ..utils import helper`)

### Phase 2 — Dependency Graph
- **Problem with naive approach**: Two files named `utils.py` in different packages would cross-wire
- **Solution**: Built full module-path index — converts `/root/a/b/c.py` → `"a.b.c"` for resolution
- Separate `FileGraph` (file → [files it imports]) and `SymbolGraph` (caller → [callees])
- DFS over FileGraph expands seed files to structurally related files (max depth = 2)

### Phase 3 — Chunking
- One chunk per function, class, and method (not whole-file chunks)
- Each chunk has a **stable ID**: `rel/path::QualName@startline` — survives file moves
- Each chunk has a **SHA-256 content hash** for incremental reindexing
- Methods get a `# (in class ParentName)` header so the embedder has context

### Phase 4 — Vector Store
- **Problem**: `IndexIDMap2` in FAISS segfaults on macOS faiss-cpu 1.13.2
- **Solution**: Plain `IndexFlatIP` + Python-managed ID mapping (`_row_to_id: list[int]`, `_id_to_row: dict[int,int]`)
- Logical delete: zeroes out the vector row, marks it deleted, filters from search results
- Persists as JSON manifest + `.npz` embeddings (no pickle — security risk)

### Phase 5 — Hybrid Retrieval
```python
# The exact formula (core/pipeline/codebase_pipeline.py)
hybrid_score = 0.7 * semantic_score + 0.3 * graph_score

# graph_score = 1.0 / (1.0 + depth)
# depth = how many hops in the import graph from seed files to this file
```

### Phase 6 — Cross-Encoder Reranking
- Take top-20 from hybrid scoring (the "rerank pool")
- Cross-encoder scores each `(query, chunk)` pair independently — more accurate than bi-encoder
- Return top-8 for prompt construction
- **Why not bi-encoder for reranking?** Bi-encoders embed query and chunk separately then dot-product. Cross-encoders see both together — much better at understanding relevance but slower. Acceptable at top-20 pool size.

### Phase 7 — Incremental Indexing
- On startup: SHA-256 hash every source file
- Compare against stored hashes in `metadata.json`
- Only re-embed files whose hash changed — unchanged files reuse stored vectors
- Result: ~15s first run, <1s subsequent runs

---

## 6. BENCHMARK RESULTS

### Setup
- **Repo**: pallets/flask (real open-source repo, not a toy)
- **Questions**: 30 hand-labeled natural language questions
- **Metric**: Recall@K (did the right file appear in top-K?), MRR (how high did it rank?)
- **Top-K**: 10

### Results Table
```
strategy       recall@5   recall@10   mrr    p50_ms   p95_ms
─────────────────────────────────────────────────────────────
bm25           0.500      0.633       0.252    0.5      0.8
semantic       0.633      0.667       0.437    6.8     18.9
hybrid         0.633      0.667       0.437    4.5      6.2
hybrid+rerank  0.667      0.667       0.536  123.6    143.0
```

### What These Numbers Mean
- **hybrid+rerank vs bm25**: +33% Recall@5, +112% MRR — graph + reranking dramatically helps
- **hybrid+rerank vs semantic**: +23% MRR — the cross-encoder reranker adds real signal
- **hybrid vs semantic**: Same accuracy, 3x faster p95 latency — graph traversal is efficient
- **p95 = 143ms**: Acceptable for interactive use (under 200ms feels instant to humans)

### How to Reproduce
```bash
git clone --depth=1 https://github.com/pallets/flask /tmp/flask_eval
cd /path/to/ai-code-agent
python -m evaluation.run_eval --repo /tmp/flask_eval --eval flask
```

---

## 7. KEY ENGINEERING CHALLENGES SOLVED

### Challenge 1: FAISS Segfault on macOS
- **Problem**: `IndexIDMap2(IndexFlatIP)` + `IDSelectorBatch` causes exit code 139 (SIGSEGV) on macOS faiss-cpu 1.13.2
- **Root cause**: Native C++ code in FAISS ID map has a memory bug on ARM macOS
- **Solution**: Replaced with plain `IndexFlatIP` and Python-managed ID mapping. `_row_to_id: list[int]` maps FAISS row numbers to stable integer IDs. `_id_to_row: dict[int,int]` maps back. `_deleted: set[int]` tracks logical deletes. Zero native FAISS ID-map code paths.
- **Lesson**: When a native library segfaults, push the problematic logic into Python.

### Challenge 2: Import Resolution Collision
- **Problem**: Two files `auth/utils.py` and `db/utils.py` both match `from utils import helper` — naive basename matching cross-wires them
- **Solution**: Built `_module_to_file: dict[str, str]` index using full module paths (`"auth.utils"` → `/root/auth/utils.py`). Resolution converts import string to module path, then looks up the index.
- **Lesson**: String matching on filenames is always wrong. Use full qualified paths.

### Challenge 3: Caller Tracking in AST
- **Problem**: Need to know that `Calculator.square` calls `self.multiply`, not just that `multiply` is called somewhere
- **Solution**: Scope stack `_scope: list[tuple[str, str]]` pushed on every `visit_FunctionDef` and `visit_ClassDef`. When a `Call` node is visited, the top of the stack gives `caller_qualname`.
- **Lesson**: AST visitor state must be explicit. Never rely on implicit walk order.

### Challenge 4: Test Isolation with ML Models
- **Problem**: Loading `sentence-transformers` twice in the same pytest process corrupts memory (model weights loaded twice)
- **Solution**: Session-scoped fixtures (`shared_embedder`, `shared_ranker`) load models once. Dependency injection (`embedder=`, `cross_ranker=` kwargs on `CodebasePipeline`) passes the shared instances into each test.
- **Lesson**: ML model loading is expensive and stateful. Session scope + DI is the correct pattern.

### Challenge 5: Frozen Dataclass in Tests
- **Problem**: `monkeypatch.setattr(pipeline.config.storage, "directory", str(tmp_path))` fails — `StorageConfig` is frozen
- **Solution**: Construct the full `Config(storage=StorageConfig(directory=str(tmp_path)))` upfront and pass to the pipeline constructor
- **Lesson**: Frozen dataclasses are immutable by design. The fix is to build the right config, not patch after construction.

---

## 8. EVERY QUESTION A GOOGLE INTERVIEWER WILL ASK

---

### Q: What problem does this project solve?

**A:** General AI chatbots like ChatGPT can't answer questions about your private codebase — they've never seen it. Even if you paste code, you can't fit a 50,000-line repo into a context window, and the LLM gets confused by irrelevant code. My project indexes any Python repo locally and lets you ask plain-English questions. It automatically finds the right code using semantic search + graph traversal, reranks the results, and generates a grounded answer using a local LLM — no API cost, no data leaves your machine.

---

### Q: Explain the retrieval pipeline end to end.

**A:** Five steps:
1. **Index**: Parse repo with Python AST → extract symbols → build call/import graph → chunk code by function/class/method → embed each chunk with MiniLM-L6-v2 (384-dim) → store in FAISS IndexFlatIP.
2. **Embed query**: Same MiniLM model embeds the user's question into a 384-dim vector.
3. **Semantic search**: FAISS returns top-20 chunks by cosine similarity.
4. **Graph expansion**: DFS on the import graph from seed files (depth ≤ 2) adds structurally related files. Combined with semantic score: `0.7 × semantic + 0.3 × graph_score`.
5. **Rerank + generate**: Cross-encoder scores each (query, chunk) pair from the top-20 pool. Top-8 go into the LLM prompt. Llama3 generates an answer grounded only in those chunks.

---

### Q: Why did you use a cross-encoder for reranking instead of just using the bi-encoder score?

**A:** Bi-encoders (like MiniLM) encode the query and the chunk independently and take the dot product. They're fast but lose information — the model never sees query and chunk together. Cross-encoders take the concatenated `[query, chunk]` as a single input and output a relevance score. They're much more accurate because they can attend to interactions between query tokens and chunk tokens. The tradeoff is speed — cross-encoders are ~10x slower — but we only run them on 20 candidates (the rerank pool), so it's acceptable: p95 latency is 143ms.

---

### Q: Why FAISS instead of a vector database like Pinecone or Weaviate?

**A:** Three reasons. First, privacy — this project is designed to run fully locally on private codebases. Pinecone and Weaviate require sending your code to their servers. Second, cost — FAISS is free and runs in-process with no network overhead. Third, simplicity — for a single-repo use case with up to ~5,000 chunks, `IndexFlatIP` is fast enough (sub-millisecond search). A managed vector DB makes sense when you have millions of vectors, multi-tenancy, or distributed teams.

---

### Q: What is MRR and why did you choose it as a metric?

**A:** MRR stands for Mean Reciprocal Rank. For each question, the reciprocal rank is 1/position of the first correct result. If the right file is rank 1, RR = 1.0. Rank 2 → 0.5. Rank 5 → 0.2. MRR is the mean across all questions. I chose it because it rewards finding the right file *high* in the list, not just somewhere in the top-10. Recall@K tells you if the file was found at all — MRR tells you if it was found first. For a RAG system where the LLM prompt has limited space, rank matters more than presence.

---

### Q: What is RAG and how did you implement it?

**A:** RAG (Retrieval Augmented Generation) is a pattern where instead of asking an LLM to generate from memory, you first retrieve relevant context from an external store and inject it into the prompt. My implementation: (1) retrieve the top-8 most relevant code chunks for the query, (2) build a prompt that says "here is the code context, answer the question using only this context", (3) pass to Llama3 via Ollama. The LLM is constrained to the retrieved chunks — it cannot hallucinate functions that don't exist in the repo.

---

### Q: How does the dependency graph work?

**A:** I build two separate graphs from the AST: a `FileGraph` (file A imports file B → edge A→B) and a `SymbolGraph` (function `compute` calls function `add` → edge `compute`→`add`). For retrieval, I use the FileGraph. When semantic search returns seed files, I run DFS (max depth 2) on the FileGraph to find files that import or are imported by the seed files. These get a graph score of `1 / (1 + depth)` — direct importers get 0.5, depth-2 neighbors get 0.33. This score is weighted 0.3 in the hybrid formula.

---

### Q: Why did you separate FileGraph and SymbolGraph?

**A:** In my first version I used one dict with mixed keys — sometimes file paths, sometimes symbol names. This was a bug. `from utils import helper` would match ANY file named `utils.py` in the repo, including `auth/utils.py` and `db/utils.py`. By building a module-path index (`"auth.utils"` → `/root/auth/utils.py`), I resolve imports correctly using full qualified paths. Separating the graphs also makes the code clearer — file-level and symbol-level relationships are fundamentally different things.

---

### Q: What is incremental indexing and how does it work?

**A:** On first run, I hash every source file with SHA-256 and store the hashes in `metadata.json`. On subsequent runs, I re-hash the files and compare. Files whose hash didn't change reuse their stored embeddings from the `.npz` file. Only changed files get re-parsed, re-chunked, and re-embedded. The changed file's old vectors are logically deleted from FAISS (zeroed out, marked in `_deleted` set), and new vectors are added. This reduces a 15-second full index to under 1 second for typical edits.

---

### Q: Why did you use JSON + .npz instead of pickle for persistence?

**A:** Pickle is a security risk — a malicious `.pkl` file can execute arbitrary code when loaded. Since this tool indexes third-party repos, a compromised repo could include a crafted storage file. JSON is human-readable, safe, and has no execution risk. NumPy's `.npz` is a safe binary format for the embedding arrays. This is a deliberate security decision documented in the commit history.

---

### Q: How did you handle the FAISS segfault?

**A:** `IndexIDMap2` in faiss-cpu 1.13.2 on macOS ARM segfaults when you call `remove_ids` with `IDSelectorBatch`. This is a bug in the native C++ code. My fix: stop using `IndexIDMap2` entirely. Use plain `IndexFlatIP` and manage the ID-to-row mapping in Python. I maintain `_row_to_id: list[int]` and `_id_to_row: dict[int,int]` manually. Logical deletes zero out the vector and add the ID to `_deleted: set[int]`, then filter deleted IDs out of search results. No native FAISS ID-map code is ever called.

---

### Q: How does the AST parser track which function calls which?

**A:** I use `ast.NodeVisitor` with an explicit scope stack: `_scope: list[tuple[str, str]]`. Every time I enter a `FunctionDef`, `AsyncFunctionDef`, or `ClassDef`, I push `(kind, qualified_name)` onto the stack. Every time I encounter a `Call` node, I read the top of the stack to get `caller_qualname`. On exit from the def, I pop the stack. This gives me every call site with its exact enclosing scope — e.g., `Calculator.square` calls `self.multiply`. Without the scope stack, I'd only know that `multiply` is called somewhere in the file, not who calls it.

---

### Q: How does the CLI route commands vs. free-form questions?

**A:** There's a `COMMANDS` tuple of `Command` dataclasses, each with a prefix string. The `ask()` method checks if the query lowercased starts with any command prefix. If it matches AND the argument after the prefix has no spaces (i.e., it's a symbol name, not a sentence), it dispatches to the corresponding pipeline method. Otherwise it falls through to `generate_answer()` for RAG. This catches the edge case of "explain how X works" — that starts with "explain " but is a question, not a symbol name, so it correctly goes to RAG.

---

### Q: What would you improve if you had more time?

**A:** Four things:
1. **Multi-language**: Replace Python's `ast` module with tree-sitter to support TypeScript, Go, Java — the same retrieval pipeline works, only the parser changes.
2. **Persistent daemon**: Right now the index is per-session. Add a background process that watches for file changes and updates the index incrementally — similar to an LSP server.
3. **Larger eval set**: 30 questions is a start. I'd hand-label 200+ questions across 5 repos (Flask, Django, FastAPI, SQLAlchemy, Celery) for more statistical confidence.
4. **HNSW index**: Replace `IndexFlatIP` (exact search, O(n)) with HNSW (approximate, O(log n)) for repos with 100K+ chunks.

---

### Q: What design patterns did you use?

**A:** 
- **Dependency Injection**: `CodebasePipeline(embedder=..., cross_ranker=...)` — models are injected, not constructed internally. This allows tests to share model instances across the session.
- **Command pattern**: `COMMANDS: tuple[Command, ...]` registry routes structured commands. Adding a new command is one line in the tuple, not an if-else chain.
- **Strategy pattern**: The eval harness defines four retrieval strategies as functions, each taking `(pipeline, query, top_k)` and returning file paths. Adding a new strategy is one tuple entry.
- **Visitor pattern**: `_SymbolVisitor(ast.NodeVisitor)` — the standard Python pattern for AST traversal.
- **Dataclass immutability**: All config objects are frozen dataclasses. Mutation is a bug caught at runtime.

---

### Q: How did you test this project?

**A:** Three layers:
1. **Unit tests**: `test_ast_parser.py` — verifies the parser extracts correct symbols, call sites, imports from a tiny 2-file repo. `test_chunker.py` — verifies chunk content, stable IDs, content hashes.
2. **Integration tests**: `test_pipeline.py` — indexes a real (tiny) repo, verifies vector store is built, `retrieve()` finds relevant chunks, `generate_answer()` returns a dict with sources, incremental reindex picks up a new function.
3. **Eval harness**: `evaluation/run_eval.py` — 4 strategies, 30 questions on Flask, measures Recall@5, Recall@10, MRR, latency. This is end-to-end correctness testing.
Total: 37 tests, 76% code coverage.

---

### Q: What is the system design if this needed to serve 1000 users?

**A:** Current design is single-user local. To scale:
- **Shared index service**: One process maintains the FAISS index, serves gRPC for retrieval queries. Multiple API workers call it.
- **Switch to HNSW**: `IndexFlatIP` is O(n) — replace with HNSW for O(log n) at scale.
- **Replace Ollama with vLLM**: vLLM supports batched inference and continuous batching. Llama3 on a single A100 can handle ~100 requests/second.
- **Async everything**: The FastAPI layer is already async. Add a task queue (Celery/Redis) for indexing jobs.
- **Cache embeddings**: Identical queries get the same embedding — add Redis cache on `embed_query()`.
- **Multi-repo**: One FAISS index per repo, a routing layer maps user → their repo's index.

---

### Q: Why Llama3 specifically?

**A:** It was the best freely-available model that runs on a MacBook at inference time when the project was built. The model choice is actually configurable — `--model` flag or `CODE_AGENT_MODEL` env var. Any Ollama-compatible model works. I chose Llama3 because it has strong instruction-following (needed for grounded generation) and a 4.7GB size that fits in 8GB RAM without quantization artifacts. For production I'd evaluate Mistral, Phi-3, or CodeLlama.

---

## 9. NUMBERS TO MEMORIZE

| Fact | Number |
|------|--------|
| Flask eval: chunks indexed | 1,624 chunks, 83 files |
| Flask eval: questions | 30 hand-labeled |
| hybrid+rerank Recall@5 | **0.667** |
| hybrid+rerank MRR | **0.536** |
| BM25 baseline Recall@5 | 0.500 |
| BM25 baseline MRR | 0.252 |
| MRR improvement over BM25 | **+112%** |
| p95 latency (hybrid+rerank) | **143ms** |
| Embedding model dimensions | 384 |
| Hybrid score weights | 0.7 semantic + 0.3 graph |
| Rerank pool size | 20 candidates |
| Final chunks to LLM | 8 chunks |
| Graph DFS max depth | 2 hops |
| Self-repo chunks | 220 chunks, 36 files |
| Test count | 37 tests |
| Code coverage | 76% |

---

## 10. WHAT NOT TO SAY

| Don't Say | Say Instead |
|-----------|-------------|
| "It uses AI" | "It uses MiniLM-L6-v2 for embeddings and Llama3 for generation" |
| "It works great" | "Recall@5 = 0.667, MRR = 0.536 on 30 Flask questions" |
| "It's like ChatGPT" | "It's a RAG system — retrieval-augmented, not generative from memory" |
| "I used some ML library" | "sentence-transformers 5.2.0, FAISS 1.13.2, cross-encoder ms-marco-MiniLM-L-6-v2" |
| "It was hard" | Describe the specific problem and specific solution |
| "It only works for Python" | "Scoped to Python for depth; tree-sitter is the natural extension path" |
| "I don't know why I chose X" | Every choice has a reason — see the table in Section 4 |

---

## 11. HOW TO DEMO (2-Minute Live Demo)

```bash
# Terminal 1 (keep running)
ollama serve

# Terminal 2
cd /path/to/ai-code-agent
python cli.py --repo /tmp/flask_eval
```

**Type these in order:**
```
>>> how does Flask handle sessions?
>>> explain SecureCookieSession
>>> how does url_for generate URLs?
>>> where is the test client defined?
```

**Then exit and run the benchmark:**
```bash
python -m evaluation.run_eval --repo /tmp/flask_eval --eval flask
```

Point to the table and say:
> "This is my evaluation harness. Four strategies — BM25 baseline, semantic only,
> hybrid, and hybrid+rerank. My best strategy improves MRR by 112% over BM25
> on 30 hand-labeled questions from the Flask codebase."
