"""Correctness and failure tests for the educational byte-level BPE."""

import json
from pathlib import Path

import pytest

from forgellm.tokenization.bpe import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    ByteBPETokenizer,
    Merge,
    TokenizerError,
)
from forgellm.tokenization.config import TokenizerConfig


def _naive_encode(tokenizer: ByteBPETokenizer, text: str) -> list[int]:
    sequence = [byte + 4 for byte in text.encode("utf-8")]
    for merge in tokenizer.merges:
        output: list[int] = []
        index = 0
        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == merge.left_id
                and sequence[index + 1] == merge.right_id
            ):
                output.append(merge.new_id)
                index += 2
            else:
                output.append(sequence[index])
                index += 1
        sequence = output
    return sequence


def test_heap_encoder_exactly_matches_naive_rank_replay() -> None:
    texts = ["banana bandana " * 5, "你好 banana🙂\n" * 4]
    tokenizer = ByteBPETokenizer.train(TokenizerConfig(300, 2), texts)

    for text in [*texts, "bandana banana", "🙂你好 banana", "", "aaaaabaaaa"]:
        assert tokenizer.encode(text) == _naive_encode(tokenizer, text)


def _config(*, vocab_size: int = 280, min_frequency: int = 1) -> TokenizerConfig:
    return TokenizerConfig(vocab_size=vocab_size, min_pair_frequency=min_frequency)


def test_manual_pair_counts_and_ranked_merges_are_explainable() -> None:
    tokenizer = ByteBPETokenizer.train(_config(vocab_size=262), ["abab", "abab"])

    assert tokenizer.merges == (
        Merge(left_id=101, right_id=102, new_id=260),
        Merge(left_id=260, right_id=260, new_id=261),
    )
    assert tokenizer.encode("abab") == [261]
    assert tokenizer.token_bytes(261) == b"abab"


def test_training_is_independent_of_document_order() -> None:
    texts = ["banana bandana", "字节字节", "🙂 banana"]

    first = ByteBPETokenizer.train(_config(), texts)
    second = ByteBPETokenizer.train(_config(), reversed(texts))

    assert first.fingerprint() == second.fingerprint()
    assert first.merges == second.merges


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain English",
        "中文分词",
        "🙂🚀🧠",
        "e\u0301 and é",
        "  spaces\tand\nnewlines  ",
    ],
)
def test_round_trip_preserves_exact_utf8_text(text: str) -> None:
    tokenizer = ByteBPETokenizer.train(
        _config(vocab_size=300), ["English 中文 🙂 repeat repeat", text]
    )

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_token_contract_is_fixed() -> None:
    tokenizer = ByteBPETokenizer.train(_config(), ["hello hello"])
    encoded = tokenizer.encode("hello", add_bos=True, add_eos=True)

    assert (PAD_ID, BOS_ID, EOS_ID, UNK_ID) == (0, 1, 2, 3)
    assert encoded[0] == BOS_ID
    assert encoded[-1] == EOS_ID
    assert tokenizer.decode([PAD_ID, *encoded, UNK_ID]) == "hello"
    assert tokenizer.decode([BOS_ID, EOS_ID], skip_special_tokens=False) == "<bos><eos>"


def test_minimum_frequency_can_stop_before_vocab_budget() -> None:
    tokenizer = ByteBPETokenizer.train(_config(vocab_size=300, min_frequency=2), ["ab"])

    assert tokenizer.vocab_size == 260
    assert tokenizer.merge_count == 0


def test_save_load_preserves_model_and_refuses_overwrite(tmp_path: Path) -> None:
    original = ByteBPETokenizer.train(_config(), ["save and load save and load"])
    path = tmp_path / "tokenizer.json"
    original.save(path)

    loaded = ByteBPETokenizer.load(path)

    assert loaded.fingerprint() == original.fingerprint()
    assert loaded.encode("save and load") == original.encode("save and load")
    with pytest.raises(FileExistsError):
        original.save(path)


def test_corrupt_model_hash_and_structure_fail(tmp_path: Path) -> None:
    tokenizer = ByteBPETokenizer.train(_config(), ["corrupt corrupt"])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    model = json.loads(path.read_text(encoding="utf-8"))
    model["model_sha256"] = "0" * 64
    path.write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(TokenizerError, match="SHA-256 mismatch"):
        ByteBPETokenizer.load(path)


def test_invalid_token_id_and_partial_utf8_fail() -> None:
    tokenizer = ByteBPETokenizer.train(_config(), ["中文"])

    with pytest.raises(TokenizerError, match="out of range"):
        tokenizer.decode([99999])
    with pytest.raises(TokenizerError, match="not valid UTF-8"):
        tokenizer.decode([0xE4 + 4])
