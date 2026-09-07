"""Conservative secret redaction for durable diagnostic artifacts."""

from __future__ import annotations

import re
from typing import Any

_PRIVATE_KEY = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?-----END(?: [A-Z0-9]+)? PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
_ASSIGNMENT = re.compile(
    r"\b(api[_-]?key|access[_-]?token|authorization|password|passwd|secret|client[_-]?secret)"
    r"\b\s*([=:])\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{8,}\b", re.IGNORECASE)
_KNOWN_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
)
_SOURCE_QUOTED_ASSIGNMENT = re.compile(
    r"(?P<name>\b(?:api[_-]?key|access[_-]?token|authorization|password|passwd|"
    r"secret|client[_-]?secret)\b)(?P<separator>\s*[=:]\s*)(?P<quote>['\"])"
    r"(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)


def redact_text(value: str) -> str:
    """Mask common credential forms while keeping surrounding evidence readable."""
    value = _PRIVATE_KEY.sub("[REDACTED PRIVATE KEY]", value)
    value = _ASSIGNMENT.sub(r"\1\2[REDACTED]", value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    return _KNOWN_TOKEN.sub("[REDACTED TOKEN]", value)


def redact_source_text(value: str) -> str:
    """Redact literal credentials without rewriting ordinary source expressions.

    Source packages may legitimately contain names such as ``secret`` or
    ``api_key``. Replacing a whole assignment expression makes a review
    package syntactically misleading, so this variant preserves identifiers,
    operators and quotes while masking only literal secret values plus known
    token forms.
    """

    def replace_literal(match: re.Match[str]) -> str:
        return (
            f"{match.group('name')}{match.group('separator')}"
            f"{match.group('quote')}[REDACTED]{match.group('quote')}"
        )

    value = _PRIVATE_KEY.sub("[REDACTED PRIVATE KEY]", value)
    value = _SOURCE_QUOTED_ASSIGNMENT.sub(replace_literal, value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    return _KNOWN_TOKEN.sub("[REDACTED TOKEN]", value)


def redact_json_value(value: Any) -> Any:
    """Recursively redact a JSON-compatible value without mutating its input."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(key): redact_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [redact_json_value(item) for item in value]
    return value
