"""HTTP client for a local Ollama server.

Provides synchronous and async generation, with explicit timeouts and a
small retry budget for transient 5xx / network errors. Streaming is also
supported (token-by-token) for the FastAPI /ask_stream endpoint.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator

import httpx

from core.config import DEFAULT_CONFIG, LLMConfig

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Generic LLM call failure (network, timeout, server error, bad payload)."""


class OllamaUnavailable(LLMError):
    """The Ollama server is unreachable."""


class OllamaClient:
    def __init__(
        self,
        model: str | None = None,
        config: LLMConfig | None = None,
    ):
        cfg = config or DEFAULT_CONFIG.llm
        if model:
            cfg = LLMConfig(
                model=model,
                base_url=cfg.base_url,
                timeout_s=cfg.timeout_s,
                max_retries=cfg.max_retries,
                temperature=cfg.temperature,
            )
        self.config = cfg
        self.url = f"{cfg.base_url.rstrip('/')}/api/generate"
        self._client: httpx.Client | None = None
        self._aclient: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _sync(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.config.timeout_s)
        return self._client

    def _async(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self.config.timeout_s)
        return self._aclient

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    # ------------------------------------------------------------------ #
    # Generate
    # ------------------------------------------------------------------ #

    def _payload(self, prompt: str, *, stream: bool) -> dict[str, object]:
        return {
            "model": self.config.model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": self.config.temperature},
        }

    def generate(self, prompt: str) -> str:
        payload = self._payload(prompt, stream=False)
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._sync().post(self.url, json=payload)
            except httpx.ConnectError as exc:
                raise OllamaUnavailable(
                    f"could not reach Ollama at {self.config.base_url}: {exc}"
                ) from exc
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("ollama request failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(min(2**attempt, 5))
                continue

            if resp.status_code >= 500:
                last_exc = LLMError(f"server {resp.status_code}: {resp.text[:200]}")
                logger.warning("ollama 5xx (attempt %d): %s", attempt + 1, resp.status_code)
                time.sleep(min(2**attempt, 5))
                continue
            if resp.status_code != 200:
                raise LLMError(f"ollama error {resp.status_code}: {resp.text[:500]}")

            try:
                return str(resp.json()["response"])
            except (KeyError, ValueError) as exc:
                raise LLMError(f"unexpected ollama payload: {exc}") from exc

        assert last_exc is not None
        raise LLMError(f"ollama failed after retries: {last_exc}")

    def stream(self, prompt: str) -> Iterator[str]:
        payload = self._payload(prompt, stream=True)
        try:
            with self._sync().stream("POST", self.url, json=payload) as resp:
                if resp.status_code != 200:
                    raise LLMError(f"ollama error {resp.status_code}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response")
                    if token:
                        yield str(token)
                    if chunk.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(
                f"could not reach Ollama at {self.config.base_url}: {exc}"
            ) from exc

    async def agenerate(self, prompt: str) -> str:
        payload = self._payload(prompt, stream=False)
        try:
            resp = await self._async().post(self.url, json=payload)
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(
                f"could not reach Ollama at {self.config.base_url}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise LLMError(f"ollama error {resp.status_code}: {resp.text[:500]}")
        try:
            return str(resp.json()["response"])
        except (KeyError, ValueError) as exc:
            raise LLMError(f"unexpected ollama payload: {exc}") from exc

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        payload = self._payload(prompt, stream=True)
        try:
            async with self._async().stream("POST", self.url, json=payload) as resp:
                if resp.status_code != 200:
                    raise LLMError(f"ollama error {resp.status_code}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response")
                    if token:
                        yield str(token)
                    if chunk.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(
                f"could not reach Ollama at {self.config.base_url}: {exc}"
            ) from exc

    async def health(self) -> bool:
        try:
            resp = await self._async().get(f"{self.config.base_url.rstrip('/')}/api/tags")
        except httpx.HTTPError:
            return False
        return resp.status_code == 200
