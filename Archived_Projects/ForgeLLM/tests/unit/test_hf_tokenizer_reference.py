"""Contract tests for the optional mature tokenizer reference."""

import pytest

pytest.importorskip("tokenizers")

from forgellm.tokenization.config import TokenizerConfig  # noqa: E402
from forgellm.tokenization.hf_reference import HuggingFaceByteBPE  # noqa: E402


def test_reference_has_fixed_special_ids_and_exact_round_trip() -> None:
    texts = ["hello hello", "中文 中文", "🙂 byte byte", "e\u0301 and é"]
    reference = HuggingFaceByteBPE.train(
        TokenizerConfig(vocab_size=300, min_pair_frequency=1), texts
    )

    assert reference.special_token_ids() == {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }
    assert reference.unknown_token_id == 3
    for text in [*texts, "", "  exact spaces  "]:
        assert reference.decode(reference.encode(text)) == text
