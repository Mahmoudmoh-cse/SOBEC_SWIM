import pytest

from app.ai.parser import parse_ai_response
from app.ai.providers import AIProviderConfigError, LocalAIProvider
from app.ai.schemas import AIProviderRequest
from app.ai.service import select_ai_provider
from app.core.config import Settings


def test_local_provider_returns_deterministic_output() -> None:
    provider = LocalAIProvider()
    response = provider.generate(
        AIProviderRequest(
            output_type="technique_summary",
            prompt_text="test",
            context={
                "swimmer_name": "Maya",
                "technique_score": 82,
                "top_fault": "Dropped elbow",
                "time_gain_possible": 0.3,
            },
        )
    )

    parsed = parse_ai_response(response.raw_response)
    assert response.provider == "local"
    assert response.model == "local-deterministic-v1"
    assert "Maya" in parsed["summary"]
    assert parsed["recommendations"]


def test_provider_selection_local() -> None:
    provider = select_ai_provider(Settings(ai_provider="local", anthropic_api_key=None))
    assert isinstance(provider, LocalAIProvider)


def test_provider_selection_hybrid_falls_back_without_api_key() -> None:
    provider = select_ai_provider(Settings(ai_provider="hybrid", anthropic_api_key=None))
    assert isinstance(provider, LocalAIProvider)


def test_provider_selection_anthropic_requires_api_key() -> None:
    with pytest.raises(AIProviderConfigError, match="ANTHROPIC_API_KEY"):
        select_ai_provider(Settings(ai_provider="anthropic", anthropic_api_key=None))


def test_parser_handles_raw_json() -> None:
    parsed = parse_ai_response(
        '{"summary":"Good work","key_findings":["A"],"recommendations":["B"],"risk_flags":[],"next_steps":["C"]}'
    )
    assert parsed["summary"] == "Good work"
    assert parsed["key_findings"] == ["A"]
    assert parsed["next_steps"] == ["C"]


def test_parser_handles_fenced_json() -> None:
    parsed = parse_ai_response(
        '```json\n{"summary":"From fence","key_findings":["A"],"recommendations":[],"risk_flags":[],"next_steps":[]}\n```'
    )
    assert parsed["summary"] == "From fence"
    assert parsed["key_findings"] == ["A"]


def test_parser_handles_plain_text_fallback() -> None:
    parsed = parse_ai_response("Plain coaching text")
    assert parsed == {
        "summary": "Plain coaching text",
        "key_findings": [],
        "recommendations": [],
        "risk_flags": [],
        "next_steps": [],
    }

