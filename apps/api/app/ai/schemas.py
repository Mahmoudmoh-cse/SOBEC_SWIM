from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXPECTED_OUTPUT_KEYS = ("summary", "key_findings", "recommendations", "risk_flags", "next_steps")


def empty_parsed_output(summary: str = "") -> dict[str, Any]:
    return {
        "summary": summary,
        "key_findings": [],
        "recommendations": [],
        "risk_flags": [],
        "next_steps": [],
    }


@dataclass(frozen=True)
class AIProviderRequest:
    output_type: str
    prompt_text: str
    context: dict[str, Any]
    model_tier: str = "fast"


@dataclass(frozen=True)
class AIProviderResult:
    provider: str
    model: str
    raw_response: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None

