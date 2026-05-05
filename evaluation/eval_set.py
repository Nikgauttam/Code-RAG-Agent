"""Self-evaluation set: questions answered by THIS repo's own code.

Each entry is (question, expected_files). expected_files are paths
relative to the repo root. A retrieval is correct if at least one of
the expected files appears in the top-k results.
"""


EVAL_SET: list[tuple[str, list[str]]] = [
    (
        "How does the AST parser extract function and class definitions?",
        ["core/parser/ast_parser.py"],
    ),
    (
        "Where are imports parsed into the symbol table?",
        ["core/parser/ast_parser.py"],
    ),
    (
        "How is the dependency graph built from the symbol table?",
        ["core/graph/dependency_graph.py"],
    ),
    (
        "Where is DFS over the dependency graph implemented?",
        ["core/graph/dependency_graph.py", "core/pipeline/codebase_pipeline.py"],
    ),
    (
        "How are code chunks created from each function and class?",
        ["core/retrieval/chunker.py"],
    ),
    (
        "Where do we encode chunks into embeddings?",
        ["core/retrieval/embedder.py"],
    ),
    (
        "How is the FAISS index built and queried?",
        ["core/retrieval/vector_store.py"],
    ),
    (
        "Where is the cross-encoder reranker defined?",
        ["core/rerank/cross_encoder_ranker.py"],
    ),
    (
        "How does the pipeline call the local LLM?",
        ["core/llm/ollama_client.py", "core/pipeline/codebase_pipeline.py"],
    ),
    (
        "Where is the hybrid scoring of semantic and graph signals?",
        ["core/pipeline/codebase_pipeline.py"],
    ),
    (
        "How is the FAISS index cached to disk and reloaded?",
        ["core/pipeline/codebase_pipeline.py", "core/retrieval/vector_store.py"],
    ),
    (
        "Where does the CLI parse the --repo argument?",
        ["cli.py"],
    ),
    (
        "Which file exposes the FastAPI /ask endpoint?",
        ["api.py"],
    ),
    (
        "Where is the CodeAgent class defined?",
        ["agent/code_agent.py"],
    ),
    (
        "How does the pipeline handle the explain command?",
        ["core/pipeline/codebase_pipeline.py", "agent/code_agent.py"],
    ),
    (
        "Where is the find-usages JSON output schema constructed?",
        ["core/pipeline/codebase_pipeline.py"],
    ),
    (
        "How does graph DFS expand seed files to related files?",
        ["core/pipeline/codebase_pipeline.py"],
    ),
    (
        "Where is the trace command implemented?",
        ["core/pipeline/codebase_pipeline.py"],
    ),
    (
        "How is the SentenceTransformer model loaded?",
        ["core/retrieval/embedder.py"],
    ),
    (
        "Where do we save and load the FAISS index file?",
        ["core/retrieval/vector_store.py", "core/pipeline/codebase_pipeline.py"],
    ),
]
