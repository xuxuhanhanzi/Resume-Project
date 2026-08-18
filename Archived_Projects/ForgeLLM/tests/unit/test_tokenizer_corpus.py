"""Tests for tokenizer JSONL corpus validation."""

import hashlib
import json
from pathlib import Path

import pytest

from forgellm.tokenization.corpus import TokenizerCorpusError, read_tokenizer_jsonl


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_reader_accepts_fixture_and_preserves_empty_text() -> None:
    repo_root = Path(__file__).parents[2]
    documents = read_tokenizer_jsonl(repo_root / "tests" / "fixtures" / "tokenizer" / "test.jsonl")

    assert len(documents) == 5
    assert documents[1].subset == "chinese"
    assert documents[3].text == ""


def test_reader_validates_optional_content_hash(tmp_path: Path) -> None:
    text = "exact text"
    path = tmp_path / "corpus.jsonl"
    _write_records(
        path,
        [
            {
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "id": "one",
                "text": text,
            }
        ],
    )

    documents = read_tokenizer_jsonl(path)

    assert documents[0].subset == "all"


@pytest.mark.parametrize(
    "records",
    [
        [{"id": "same", "text": "a"}, {"id": "same", "text": "b"}],
        [{"id": "one", "text": 3}],
        [{"extra": True, "id": "one", "text": "a"}],
        [{"content_sha256": "bad", "id": "one", "text": "a"}],
    ],
)
def test_invalid_corpus_fails(tmp_path: Path, records: list[dict[str, object]]) -> None:
    path = tmp_path / "bad.jsonl"
    _write_records(path, records)

    with pytest.raises(TokenizerCorpusError):
        read_tokenizer_jsonl(path)


def test_blank_line_fails(tmp_path: Path) -> None:
    path = tmp_path / "blank.jsonl"
    path.write_text('{"id":"one","text":"a"}\n\n', encoding="utf-8")

    with pytest.raises(TokenizerCorpusError, match="blank line"):
        read_tokenizer_jsonl(path)
