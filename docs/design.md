# CodeRAG – Context-Aware AI Code Agent

## 1. Problem Statement
Traditional LLMs cannot understand large codebases.
We aim to build a context-aware AI code agent that retrieves
relevant code using both semantic similarity and structural relationships.

## 2. System Architecture

Codebase → AST Parser → Dependency Graph → Chunking → Embedding → FAISS

User Query → Embed → Top-K Retrieval → Graph Expansion → LLM → Response

## 3. Why RAG?
LLMs cannot process entire repositories due to token limits.
RAG allows retrieving only relevant context.

## 4. Why Dependency Graph?
Semantic similarity alone misses structural relationships.
Using graph traversal (DFS/BFS) helps retrieve related modules.

## 5. Data Structures Used
- Adjacency List (Graph)
- DFS / BFS
- Min Heap (Top-K Ranking)
- HashMap (Symbol Table)

## 6. Complexity Goals
Graph traversal: O(V + E)
Top-K retrieval: O(N log K)
Embedding lookup: O(N)