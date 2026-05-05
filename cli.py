"""Interactive CLI for the AI Code Agent.

Usage:
    python -m main --repo /path/to/repo
    python cli.py --repo /path/to/repo --model llama3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from agent.code_agent import CodeAgent
from core.config import Config
from core.llm.ollama_client import LLMError

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai-code-agent",
        description="Context-aware AI code agent (RAG over a code repository).",
    )
    p.add_argument(
        "--repo",
        default=os.environ.get("CODE_AGENT_REPO"),
        help="Path to the repository to index (or set $CODE_AGENT_REPO).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("CODE_AGENT_MODEL", "llama3"),
        help="Ollama model name (default: llama3).",
    )
    p.add_argument(
        "--query",
        default=None,
        help="One-shot query: ask, print, exit (no REPL).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of pretty output.",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("CODE_AGENT_LOG", "WARNING"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _render(answer: object, *, raw_json: bool) -> None:
    if raw_json:
        if isinstance(answer, dict | list):
            print(json.dumps(answer, indent=2, default=str))
        else:
            print(answer)
        return

    if isinstance(answer, dict):
        if "answer" in answer:
            console.print(
                Panel(
                    Markdown(str(answer["answer"])),
                    title="Answer",
                    border_style="cyan",
                )
            )
            sources = answer.get("sources") or []
            if sources:
                console.print("[dim]Sources:[/dim]")
                for s in sources:
                    console.print(f"  • [cyan]{s}[/cyan]")
        else:
            console.print(
                Syntax(
                    json.dumps(answer, indent=2, default=str),
                    "json",
                    theme="monokai",
                    word_wrap=True,
                )
            )
    else:
        console.print(Panel(Markdown(str(answer)), border_style="cyan"))


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(level=args.log_level)

    if not args.repo:
        console.print("[red]error:[/red] --repo is required (or set $CODE_AGENT_REPO).")
        return 2
    if not os.path.isdir(args.repo):
        console.print(f"[red]error:[/red] repository path not found: {args.repo}")
        return 2

    agent = CodeAgent(args.repo, model=args.model, config=Config.from_env())

    with console.status("[bold cyan]Indexing codebase...[/bold cyan]"):
        agent.setup()

    if args.query:
        try:
            _render(agent.ask(args.query), raw_json=args.json)
            return 0
        except LLMError as exc:
            console.print(f"[red]LLM error:[/red] {exc}")
            return 1

    console.print(
        Panel(
            agent.help_text(),
            title="AI Code Agent — interactive",
            border_style="green",
        )
    )

    while True:
        try:
            query = console.input("[bold green]>>>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            break
        if query.lower() in {"help", "?"}:
            console.print(agent.help_text())
            continue

        try:
            answer = agent.ask(query)
        except LLMError as exc:
            console.print(f"[yellow]LLM error:[/yellow] {exc}")
            continue
        except Exception as exc:
            console.print(f"[red]error:[/red] {exc}", style="red")
            continue

        _render(answer, raw_json=args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
