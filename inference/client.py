"""
Async HTTP client for the Kaggle/vLLM inference server (OpenAI-compatible API).
The inference server URL changes every Kaggle session — update via /seturl in the bot.

Model name is resolved from the registry so config stays in one place.
"""
import json
import logging
import os
from typing import AsyncIterator
import httpx
from config import INFERENCE_URL
from inference.model_registry import get_model_config, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Resolve the active model at startup; override with MODEL_NAME env var
_model_cfg = get_model_config(os.getenv("MODEL_NAME", DEFAULT_MODEL))


class InferenceClient:
    def __init__(self, base_url: str = INFERENCE_URL):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=120.0)

    def set_url(self, url: str) -> None:
        """Update inference URL at runtime (called by bot's /seturl command)."""
        self.base_url = url.rstrip("/")
        logger.info(f"Inference URL updated to {self.base_url}")

    async def health(self) -> bool:
        """Returns True if inference server is reachable."""
        try:
            r = await self._client.get(f"{self.base_url}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
    ) -> str:
        """Single-shot chat completion. Returns the assistant's response text."""
        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": _model_cfg.name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        """Streaming chat — yields text chunks as they arrive."""
        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": _model_cfg.name,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    if content := delta.get("content"):
                        yield content

    async def aclose(self) -> None:
        await self._client.aclose()


# Module-level singleton — imported by bot and reward worker
llm = InferenceClient()
