from __future__ import annotations

import time

import httpx

from app.ai.providers.base import AIProvider, AIProviderConfigError, AIProviderError
from app.ai.schemas import AIProviderRequest, AIProviderResult


class AnthropicAIProvider(AIProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None, heavy_model: str, fast_model: str) -> None:
        if not api_key:
            raise AIProviderConfigError("ANTHROPIC_API_KEY is required when AI_PROVIDER=anthropic.")
        self.api_key = api_key
        self.heavy_model = heavy_model
        self.fast_model = fast_model

    def generate(self, request: AIProviderRequest) -> AIProviderResult:
        model = self.heavy_model if request.model_tier == "heavy" else self.fast_model
        start = time.perf_counter()
        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 900,
                    "system": (
                        "You are AquaIQ Coach. Return only valid JSON with keys: "
                        "summary, key_findings, recommendations, risk_flags, next_steps."
                    ),
                    "messages": [{"role": "user", "content": request.prompt_text}],
                },
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise AIProviderError(f"Anthropic request failed: {exc}") from exc

        raw_text = _extract_text(payload)
        usage = payload.get("usage", {})
        return AIProviderResult(
            provider=self.name,
            model=model,
            raw_response=raw_text,
            latency_ms=int((time.perf_counter() - start) * 1000),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


def _extract_text(payload: dict) -> str:
    content = payload.get("content", [])
    if not content:
        return ""
    first = content[0]
    if isinstance(first, dict):
        return str(first.get("text", ""))
    return str(first)

