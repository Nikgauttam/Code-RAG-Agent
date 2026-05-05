"""FastAPI server for the code agent.

Production basics included:

  * /healthz (liveness) and /readyz (returns ready only after indexing
    is complete and the LLM is reachable).
  * Per-request UUID logged on entry/exit with structured JSON logs.
  * Configurable CORS (defaults to "*", tighten via CODE_AGENT_CORS).
  * Streaming /ask_stream endpoint using Server-Sent Events.
  * Sync calls offloaded to a threadpool so the event loop never blocks.
  * Graceful indexing on startup with a clear "ready" gate.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.code_agent import AgentNotReady, CodeAgent
from core.config import Config

# --------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO, format="%(message)s")
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
log = structlog.get_logger("code_agent.api")


# --------------------------------------------------------------------- #
# Lifespan: index on startup
# --------------------------------------------------------------------- #

REPO_PATH = os.environ.get("CODE_AGENT_REPO")
MODEL = os.environ.get("CODE_AGENT_MODEL", "llama3")
CORS_ALLOW = [o.strip() for o in os.environ.get("CODE_AGENT_CORS", "*").split(",")]


class _AppState:
    agent: CodeAgent | None = None
    ready: bool = False
    repo_path: str | None = None
    model: str | None = None


state = _AppState()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if not REPO_PATH:
        raise RuntimeError("CODE_AGENT_REPO env var must be set to the repository path.")
    log.info("startup", repo=REPO_PATH, model=MODEL)
    state.repo_path = REPO_PATH
    state.model = MODEL
    state.agent = CodeAgent(repo_path=REPO_PATH, model=MODEL, config=Config.from_env())
    await run_in_threadpool(state.agent.setup)
    state.ready = True
    log.info("ready")
    try:
        yield
    finally:
        if state.agent is not None:
            await state.agent.pipeline.llm.aclose()
        log.info("shutdown")


app = FastAPI(title="AI Code Agent API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------- #
# Per-request middleware: id + timing
# --------------------------------------------------------------------- #


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=rid, path=request.url.path)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        log.exception("unhandled", error=str(exc))
        response = JSONResponse(
            {"error": "internal_error", "request_id": rid}, status_code=500
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = rid
    log.info("request", status=response.status_code, ms=round(elapsed_ms, 2))
    structlog.contextvars.clear_contextvars()
    return response


# --------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------- #


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


class HealthResponse(BaseModel):
    status: str
    repo: str | None
    model: str | None
    ready: bool


# --------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------- #


def _require_ready() -> CodeAgent:
    if not state.ready or state.agent is None:
        raise HTTPException(status_code=503, detail="indexing in progress")
    return state.agent


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok", repo=state.repo_path, model=state.model, ready=state.ready
    )


@app.get("/readyz")
async def readyz() -> JSONResponse:
    if not state.ready or state.agent is None:
        return JSONResponse({"ready": False}, status_code=503)
    llm_ok = await state.agent.pipeline.llm.health()
    if not llm_ok:
        return JSONResponse({"ready": False, "reason": "llm_unreachable"}, status_code=503)
    return JSONResponse({"ready": True})


# Backwards-compat health endpoint.
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return await healthz()


@app.post("/ask")
async def ask_question(request: QueryRequest) -> dict[str, object] | str:
    agent = _require_ready()
    try:
        return await run_in_threadpool(agent.ask, request.query)
    except AgentNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ask_stream")
async def ask_stream(request: QueryRequest) -> StreamingResponse:
    agent = _require_ready()

    async def event_source() -> AsyncIterator[bytes]:
        # We use the sync stream() and offload via run_in_threadpool batches.
        # For true async streaming you'd switch to OllamaClient.astream.
        loop_iter = await run_in_threadpool(lambda: list(agent.stream(request.query)))
        for token in loop_iter:
            payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/explain")
async def explain_symbol(request: QueryRequest) -> dict[str, object] | str:
    agent = _require_ready()
    return await run_in_threadpool(agent.ask, f"explain {request.query}")


@app.post("/trace")
async def trace_symbol(request: QueryRequest) -> dict[str, object] | str:
    agent = _require_ready()
    return await run_in_threadpool(agent.ask, f"trace {request.query}")


@app.post("/usages")
async def find_usages(request: QueryRequest) -> dict[str, object] | str:
    agent = _require_ready()
    return await run_in_threadpool(agent.ask, f"find usages of {request.query}")
