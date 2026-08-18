"""Tests for the tokenizer-free entropy-patching demonstration."""

from forgellm.tokenization.entropy_patching import BigramEntropyPatcher


def test_entropy_patches_are_dynamic_and_lossless() -> None:
    patcher = BigramEntropyPatcher.train(["aaaaabaaaaab", "aaaaacaaaaac"] * 8)
    text = "aaaaabZaaaaac"
    patches = patcher.patch(text, threshold_bits=3.0, max_patch_bytes=6)

    assert patcher.decode(patches) == text
    assert b"".join(patch.payload for patch in patches) == text.encode()
    assert any(patch.start == 6 for patch in patches)
    assert all(1 <= len(patch.payload) <= 6 for patch in patches)


def test_empty_text_has_no_patches() -> None:
    patcher = BigramEntropyPatcher.train(["abc"])

    assert patcher.patch("") == []
    assert patcher.decode([]) == ""
