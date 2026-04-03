from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw: dict[str, Any] = field(repr=False)


class LLMClientError(Exception):
    pass


class LLMClient:
    """Thin wrapper around the OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        max_retries: int = 3,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._http = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                start = time.perf_counter()
                response = self._http.post("/chat/completions", json=payload)
                latency = (time.perf_counter() - start) * 1000
                response.raise_for_status()
                data = response.json()
                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=data["model"],
                    prompt_tokens=data["usage"]["prompt_tokens"],
                    completion_tokens=data["usage"]["completion_tokens"],
                    latency_ms=round(latency, 2),
                    raw=data,
                )
            except (httpx.HTTPStatusError, httpx.TransportError, KeyError) as exc:
                last_error = exc
                logger.warning(
                    "llm_request_failed",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error=str(exc),
                )
                if attempt < self._max_retries:
                    time.sleep(2 ** (attempt - 1))

        raise LLMClientError(f"All {self._max_retries} attempts failed") from last_error

    def close(self) -> None:
        self._http.close()
