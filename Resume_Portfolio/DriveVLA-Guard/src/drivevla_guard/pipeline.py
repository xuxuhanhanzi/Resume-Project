from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .backends import CandidateBackend
from .config import GuardConfig
from .risk import RiskScorer
from .types import PlanResult, SceneContext, ScoredCandidate


class GuardPipeline:
    def __init__(self, backend: CandidateBackend, config: GuardConfig):
        self.backend = backend
        self.config = config
        self.scorer = RiskScorer(config.risk)
        canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8"))
        package_root = Path(__file__).resolve().parent
        for filename in ["backends.py", "pipeline.py", "risk.py", "types.py"]:
            digest.update((package_root / filename).read_bytes())
        # This identity intentionally binds resumability to both configuration
        # and the decision-critical source implementation.
        self.config_hash = digest.hexdigest()[:16]

    def plan(self, model_input: dict[str, Any], context: SceneContext) -> PlanResult:
        started = time.perf_counter()
        generation = self.config.generation
        primary_mode = generation.primary_mode
        if primary_mode not in {"fast", "slow"}:
            raise ValueError(f"generation.primary_mode must be fast or slow, got {primary_mode}")
        fast = self.backend.generate(
            model_input=model_input,
            context=context,
            mode=primary_mode,
            count=generation.fast_candidates,
            seed=generation.seed,
        )
        scored_fast = tuple(ScoredCandidate(item, self.scorer.score(item, context)) for item in fast)
        ranked_fast = tuple(sorted(scored_fast, key=lambda item: item.risk.total))
        reasons = self._route_reasons(ranked_fast, context)

        slow: list = []
        if self.config.router.enabled and reasons:
            slow = self.backend.generate(
                model_input=model_input,
                context=context,
                mode="slow",
                count=generation.slow_candidates,
                seed=generation.seed,
            )
        ranked_slow = tuple(self.scorer.rank(slow, context)) if slow else ()
        if self.config.rerank_enabled:
            selected = min((*ranked_fast, *ranked_slow), key=lambda item: item.risk.total)
        elif ranked_slow:
            selected = ranked_slow[0]
        else:
            selected = scored_fast[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        candidate_latency = sum(item.candidate.latency_ms for item in (*ranked_fast, *ranked_slow))
        return PlanResult(
            scene_id=context.scene_id,
            selected=selected,
            fast_candidates=scored_fast,
            slow_candidates=ranked_slow,
            routed_to_slow=bool(ranked_slow),
            route_reasons=tuple(reasons),
            total_latency_ms=max(elapsed_ms, candidate_latency),
            config_hash=self.config_hash,
        )

    def _route_reasons(self, ranked: tuple[ScoredCandidate, ...], context: SceneContext) -> list[str]:
        cfg = self.config.router
        reasons: list[str] = []
        if all(not item.candidate.valid or item.risk.invalid > 0 for item in ranked):
            reasons.append("all_candidates_invalid")
        best = ranked[0]
        if best.risk.total >= cfg.risk_threshold:
            reasons.append("high_risk")
        if len(ranked) > 1 and ranked[1].risk.total - best.risk.total <= cfg.margin_threshold:
            reasons.append("low_score_margin")
        if best.candidate.uncertainty is not None and best.candidate.uncertainty >= cfg.uncertainty_threshold:
            reasons.append("high_model_uncertainty")
        if cfg.route_on_missing_context and not context.has_environment:
            reasons.append("missing_environment_context")
        return reasons
