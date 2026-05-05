"""High-level facade over the pipeline.

Routes free-form queries through the QA path and dispatches structured
commands ("explain X", "trace X", "find usages of X") via a small
registry instead of brittle string-prefix chains. Adding a new command
is one decorator-free entry in the COMMANDS table.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from core.config import Config
from core.pipeline.codebase_pipeline import CodebasePipeline
from core.rerank.cross_encoder_ranker import CrossEncoderRanker
from core.retrieval.embedder import CodeEmbedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Command:
    name: str
    prefix: str  # case-insensitive trigger; argument is everything after it
    handler: str  # method name on CodebasePipeline
    description: str


COMMANDS: tuple[Command, ...] = (
    Command("explain", "explain ", "explain_symbol", "Structured explanation of a symbol"),
    Command("trace", "trace ", "trace_symbol", "Trace caller/callee chain for a symbol"),
    Command("usages", "find usages of ", "find_usages", "List definitions and usages of a symbol"),
)


class AgentNotReady(RuntimeError):
    """Raised when ask() is called before setup()."""


class CodeAgent:
    def __init__(
        self,
        repo_path: str,
        model: str | None = None,
        config: Config | None = None,
        *,
        embedder: CodeEmbedder | None = None,
        cross_ranker: CrossEncoderRanker | None = None,
    ):
        self.pipeline = CodebasePipeline(
            repo_path,
            model=model,
            config=config,
            embedder=embedder,
            cross_ranker=cross_ranker,
        )
        self.indexed = False

    def setup(self) -> None:
        self.pipeline.index_codebase()
        self.indexed = True

    def ask(self, query: str) -> str | dict[str, object]:
        if not self.indexed:
            raise AgentNotReady("Agent not initialized. Call setup() first.")

        query = query.strip()
        if not query:
            return ""

        for cmd in COMMANDS:
            if query.lower().startswith(cmd.prefix):
                argument = query[len(cmd.prefix) :].strip()
                # Only dispatch as a command if the argument looks like a
                # symbol name (no spaces, or at most one dot/underscore separator).
                # Sentences like "explain how X works" fall through to RAG.
                if argument and " " not in argument:
                    handler: Callable[[str], object] = getattr(self.pipeline, cmd.handler)
                    return handler(argument)  # type: ignore[return-value]

        return self.pipeline.generate_answer(query)

    def stream(self, query: str) -> Iterable[str]:
        if not self.indexed:
            raise AgentNotReady("Agent not initialized. Call setup() first.")
        return self.pipeline.stream_answer(query)

    @staticmethod
    def help_text() -> str:
        lines = ["Commands:"]
        for cmd in COMMANDS:
            lines.append(f"  {cmd.prefix.strip():<20}  — {cmd.description}")
        lines.append("  <free-form>           — RAG question over the codebase")
        return "\n".join(lines)
