"""Optional Hugging Face Tokenizers reference adapter for Stage 1 comparisons."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from forgellm.tokenization.config import TokenizerConfig

_SPECIAL_TOKEN_ORDER = ["<pad>", "<bos>", "<eos>", "<unk>"]


class HuggingFaceReferenceError(RuntimeError):
    """Raised when the optional reference dependency or its contract is unavailable."""


def _modules() -> tuple[Any, Any, Any, Any, Any, str]:
    try:
        tokenizers = importlib.import_module("tokenizers")
        decoders = importlib.import_module("tokenizers.decoders")
        models = importlib.import_module("tokenizers.models")
        pre_tokenizers = importlib.import_module("tokenizers.pre_tokenizers")
        trainers = importlib.import_module("tokenizers.trainers")
    except ImportError as error:
        raise HuggingFaceReferenceError(
            "optional dependency missing; install requirements-tokenizer.lock"
        ) from error
    version = getattr(tokenizers, "__version__", "unknown")
    return tokenizers, decoders, models, pre_tokenizers, trainers, str(version)


class HuggingFaceByteBPE:
    """Adapter exposing the common ForgeLLM evaluation protocol."""

    def __init__(self, tokenizer: Any, *, library_version: str) -> None:
        self._tokenizer = tokenizer
        self.library_version = library_version

    @classmethod
    def train(cls, config: TokenizerConfig, texts: Sequence[str]) -> HuggingFaceByteBPE:
        """Train ByteLevel+BPE with an explicit alphabet and no prefix-space insertion."""
        tokenizers, decoders, models, pre_tokenizers, trainers, version = _modules()
        tokenizer = tokenizers.Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            trim_offsets=True,
            use_regex=False,
        )
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_pair_frequency,
            show_progress=False,
            special_tokens=_SPECIAL_TOKEN_ORDER,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        instance = cls(tokenizer, library_version=version)
        if instance.special_token_ids() != {
            token: index for index, token in enumerate(_SPECIAL_TOKEN_ORDER)
        }:
            raise HuggingFaceReferenceError(
                "reference library did not preserve the requested special-token IDs"
            )
        return instance

    @property
    def vocab_size(self) -> int:
        """Return the trained reference vocabulary size."""
        value = self._tokenizer.get_vocab_size()
        if isinstance(value, bool) or not isinstance(value, int):
            raise HuggingFaceReferenceError("reference vocabulary size is not an integer")
        return int(value)

    @property
    def unknown_token_id(self) -> int | None:
        """Return the reference unknown-token ID."""
        value = self._tokenizer.token_to_id("<unk>")
        if value is None:
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
        raise HuggingFaceReferenceError("reference unknown-token ID is invalid")

    def special_token_ids(self) -> dict[str, int | None]:
        """Return the observable special-token mapping."""
        return {token: self._tokenizer.token_to_id(token) for token in _SPECIAL_TOKEN_ORDER}

    def encode(self, text: str) -> list[int]:
        """Encode one string without automatically adding special tokens."""
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return [int(token_id) for token_id in encoding.ids]

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode while skipping registered special tokens."""
        result = self._tokenizer.decode(list(token_ids), skip_special_tokens=True)
        if not isinstance(result, str):
            raise HuggingFaceReferenceError("reference decoder returned a non-string")
        return result

    def save(self, path: Path) -> None:
        """Save the native reference model while refusing overwrite."""
        if path.exists():
            raise FileExistsError(f"reference tokenizer already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))
