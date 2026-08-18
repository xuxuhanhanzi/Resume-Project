"""Patch reviewer for detecting common code-fix anti-patterns (P8B)."""

from __future__ import annotations

import re

_SUSPICIOUS_PATTERNS: list[tuple[str, str]] = [
    (r"=\s*set\(\)\s*$", "set cleared to empty"),
    (r"=\s*\[\]\s*$", "list cleared to empty"),
    (r"=\s*\{\}\s*$", "dict cleared to empty"),
    (r"=\s*None\b", "value set to None (potential type mismatch)"),
    (r"^\s*pass\s*$", "function body replaced with pass"),
    (r"\.clear\(\)", "collection cleared in-place"),
    (r"#\s*TODO|#\s*FIXME|#\s*HACK", "debug marker left in patch"),
]


def review_diff(diff: str) -> list[str]:
    """Return warnings for suspicious patterns in added diff lines.

    Only inspects lines starting with ``+`` (excluding ``+++`` headers).
    """
    warnings: list[str] = []
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for pattern, description in _SUSPICIOUS_PATTERNS:
            if re.search(pattern, line):
                warnings.append(f"{description}: {line.strip()}")
    return warnings
