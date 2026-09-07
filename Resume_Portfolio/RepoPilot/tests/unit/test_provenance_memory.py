from __future__ import annotations

from pathlib import Path

import pytest

from repopilot.memory.provenance import (
    ContentAddressedExtractionCache,
    MemoryCard,
    MemoryKind,
    ProvenanceValidator,
)


def _card() -> tuple[MemoryCard, str]:
    source = "user: Please use compact answers.\nassistant: Acknowledged."
    return (
        MemoryCard.create(
            source_session_id="session-1",
            source_turn_ids=("turn-1",),
            observed_at="2026-08-23",
            scope="user",
            kind=MemoryKind.PREFERENCE,
            statement="The user prefers compact answers.",
            entities=("user",),
            valid_from=None,
            valid_to=None,
            confidence=0.9,
            extraction_model="test-extractor-v1",
            source_content=source,
        ),
        source,
    )


def test_memory_card_requires_source_hash_and_turn_provenance(tmp_path: Path) -> None:
    card, source = _card()
    ProvenanceValidator().validate(
        card,
        source_sessions={"session-1": source},
        source_turn_ids={"session-1": {"turn-1", "turn-2"}},
    )
    with pytest.raises(ValueError, match="content hash"):
        ProvenanceValidator().validate(
            card,
            source_sessions={"session-1": "modified"},
            source_turn_ids={"session-1": {"turn-1"}},
        )
    cache = ContentAddressedExtractionCache(tmp_path)
    cache.put_once(source, (card,))
    assert cache.get(source) == (card,)
    with pytest.raises(ValueError, match="replace"):
        cache.put_once(source, ())
