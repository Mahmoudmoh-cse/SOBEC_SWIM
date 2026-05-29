from __future__ import annotations

import json
import re
from typing import Any

from app.ai.schemas import EXPECTED_OUTPUT_KEYS, empty_parsed_output


FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)```", re.IGNORECASE)


def parse_ai_response(raw_response: str | None) -> dict[str, Any]:
    raw = (raw_response or "").strip()
    if not raw:
        return empty_parsed_output()

    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return _normalize_output(parsed, raw)

    return empty_parsed_output(summary=raw)


def _json_candidates(raw: str) -> list[str]:
    candidates = [raw]
    fenced = FENCED_JSON_RE.search(raw)
    if fenced:
        candidates.append(fenced.group(1).strip())
    return candidates


def _normalize_output(parsed: Any, fallback_summary: str) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return empty_parsed_output(summary=fallback_summary)

    output = empty_parsed_output()
    for key in EXPECTED_OUTPUT_KEYS:
        value = parsed.get(key)
        if key == "summary":
            output[key] = str(value).strip() if value is not None else fallback_summary
        else:
            output[key] = _as_string_list(value)
    return output


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []

