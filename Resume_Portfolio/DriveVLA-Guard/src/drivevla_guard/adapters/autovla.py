from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from ..config import GenerationConfig
from ..types import Candidate, SceneContext, Trajectory


class AutoVLACandidateBackend:
    """Inference-only adapter around an initialized upstream ``models.autovla.AutoVLA``.

    Imports torch only at generation time, so the core project and tests stay CPU-light.
    Candidates are generated sequentially to cap peak KV-cache memory.
    """

    def __init__(self, model: Any, generation: GenerationConfig, dt_s: float = 0.5):
        self.model = model
        self.generation = generation
        self.dt_s = dt_s

    def generate(
        self,
        model_input: dict[str, Any],
        context: SceneContext,
        mode: str,
        count: int,
        seed: int,
    ) -> list[Candidate]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("AutoVLA generation requires the optional 'autovla' dependencies") from exc

        candidates: list[Candidate] = []
        with self._thinking_mode(mode):
            inputs = self.model.get_prompt(model_input)
            model_inputs = {
                key: value.to(self.model.device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            prompt_length = int(inputs.input_ids.shape[1])
            for index in range(count):
                candidate_seed = seed + index + (10_000 if mode == "slow" else 0)
                torch.manual_seed(candidate_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(candidate_seed)
                    torch.cuda.synchronize()
                started = time.perf_counter()
                generation_kwargs = {
                    "do_sample": count > 1,
                    "max_new_tokens": self.generation.max_new_tokens,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
                if count > 1:
                    generation_kwargs.update(
                        temperature=self.generation.temperature,
                        top_p=self.generation.top_p,
                        top_k=self.generation.top_k,
                    )
                output = self.model.vlm.generate(**model_inputs, **generation_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - started) * 1000.0

                completion = output.sequences[0, prompt_length:]
                eos_id = self.model.processor.tokenizer.eos_token_id
                if eos_id is not None:
                    eos_positions = (completion == eos_id).nonzero(as_tuple=False)
                    if len(eos_positions):
                        completion = completion[: int(eos_positions[0])]
                start = int(self.model.action_start_id)
                size = int(self.model.action_tokenizer.vocab_size)
                action_mask = (completion >= start) & (completion < start + size)
                action_ids = completion[action_mask]
                token_ids = tuple(int(value) for value in action_ids.detach().cpu().tolist())
                valid = len(token_ids) == self.generation.expected_action_tokens
                error = (
                    None
                    if valid
                    else (
                        f"expected {self.generation.expected_action_tokens} action tokens, "
                        f"got {len(token_ids)}"
                    )
                )
                if token_ids:
                    decoded = self.model.action_tokenizer.decode_token_ids_to_trajectory(action_ids.cpu())
                    decoded = np.asarray(decoded)
                    if decoded.ndim == 3:
                        decoded = decoded[0]
                    poses = decoded[1 : 1 + self.generation.expected_action_tokens]
                else:
                    poses = np.zeros((self.generation.expected_action_tokens, 3), dtype=np.float64)
                if len(poses) != self.generation.expected_action_tokens:
                    valid = False
                    error = f"decoded trajectory has {len(poses)} poses"
                    poses = self._pad_trajectory(poses, self.generation.expected_action_tokens)

                log_probability = self._sequence_log_probability(output, prompt_length)
                uncertainty = None if log_probability is None else 1.0 - math.exp(min(log_probability, 0.0))
                raw_text = self.model.processor.decode(completion.detach().cpu(), skip_special_tokens=False)
                candidates.append(
                    Candidate(
                        candidate_id=f"{mode}-{index}",
                        trajectory=Trajectory(poses, dt_s=self.dt_s),
                        mode=mode,
                        token_ids=token_ids,
                        log_probability=log_probability,
                        uncertainty=uncertainty,
                        latency_ms=latency_ms,
                        valid=valid,
                        error=error,
                        raw_text=raw_text,
                    )
                )
        return candidates

    @contextmanager
    def _thinking_mode(self, mode: str) -> Iterator[None]:
        if mode not in {"fast", "slow"}:
            raise ValueError(f"unknown thinking mode: {mode}")
        original = bool(self.model.use_cot)
        self.model.use_cot = mode == "slow"
        try:
            yield
        finally:
            self.model.use_cot = original

    def _sequence_log_probability(self, output: Any, prompt_length: int) -> float | None:
        try:
            scores = self.model.vlm.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )[0]
            if len(scores) == 0:
                return None
            return float(scores.mean().detach().cpu())
        except (AttributeError, RuntimeError, IndexError, TypeError):
            return None

    @staticmethod
    def _pad_trajectory(poses: np.ndarray, length: int) -> np.ndarray:
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        if len(poses) >= length:
            return poses[:length]
        last = poses[-1] if len(poses) else np.zeros(3, dtype=np.float64)
        return np.vstack([poses, np.repeat(last[None, :], length - len(poses), axis=0)])
