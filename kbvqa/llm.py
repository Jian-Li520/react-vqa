"""OpenAI-compatible chat completions client."""

from __future__ import annotations

import os
from typing import Any

import httpx


class LLMError(RuntimeError):
    """Raised when an LLM request fails."""


class LLMConfigurationError(LLMError):
    """Raised when required LLM configuration is missing."""


class LLMClient:
    """Minimal OpenAI-compatible chat completions client."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ):
        if not api_key:
            raise LLMConfigurationError("KBVQA_API_KEY is required.")
        if not model:
            raise LLMConfigurationError("KBVQA_MODEL is required.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls(
            api_key=os.getenv("KBVQA_API_KEY", ""),
            model=os.getenv("KBVQA_MODEL", ""),
            base_url=os.getenv("KBVQA_BASE_URL", "https://api.openai.com/v1"),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        close_client = self._client is None
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise LLMError(
                f"LLM request failed with HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        finally:
            if close_client:
                client.close()

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {data}") from exc
        if not isinstance(content, str):
            raise LLMError(f"Unexpected LLM message content: {content!r}")
        return content
