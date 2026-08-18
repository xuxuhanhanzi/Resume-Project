"""Auditable fixed, rule, model, and hybrid task routing with cascade fallback."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import StrEnum

from repopilot.core.contracts import Message, ModelRequest
from repopilot.providers.base import ModelProvider
from repopilot.task import PublicTaskSpec


class RoutingMode(StrEnum):
    """Supported routing baselines."""

    FIXED = "fixed"
    RULE = "rule"
    MODEL = "model"
    HYBRID = "hybrid"


class RouteTarget(StrEnum):
    """Named runtime configurations selected by a router."""

    STANDARD = "standard"
    ESCALATED = "escalated"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """A durable route choice with its complete fallback history."""

    target: RouteTarget
    mode: RoutingMode
    reason: str
    confidence: float
    cascade: tuple[str, ...] = ()


class TaskRouter:
    """Select a runtime target without exposing tools or private evaluator data."""

    _ESCALATION_TERMS = frozenset(
        {
            "architecture",
            "concurrency",
            "deadlock",
            "flaky",
            "migration",
            "performance",
            "race condition",
            "security",
            "跨模块",
            "并发",
            "性能",
            "架构",
            "迁移",
            "安全",
        }
    )

    def __init__(
        self,
        mode: RoutingMode = RoutingMode.RULE,
        *,
        fixed_target: RouteTarget = RouteTarget.STANDARD,
        model_timeout_seconds: float = 15.0,
        hybrid_rule_threshold: float = 0.8,
    ) -> None:
        if model_timeout_seconds <= 0:
            raise ValueError("model_timeout_seconds must be positive")
        if not 0 <= hybrid_rule_threshold <= 1:
            raise ValueError("hybrid_rule_threshold must be in [0, 1]")
        self.mode = mode
        self.fixed_target = fixed_target
        self.model_timeout_seconds = model_timeout_seconds
        self.hybrid_rule_threshold = hybrid_rule_threshold

    async def route(
        self,
        task: PublicTaskSpec,
        *,
        provider: ModelProvider | None = None,
    ) -> RoutingDecision:
        """Route a public task and preserve every cascade decision."""

        if self.mode is RoutingMode.FIXED:
            return RoutingDecision(
                self.fixed_target,
                self.mode,
                "fixed baseline",
                1.0,
                (f"fixed:{self.fixed_target.value}",),
            )

        rule = self._rule_decision(task)
        if self.mode is RoutingMode.RULE:
            return rule

        if self.mode is RoutingMode.HYBRID and rule.confidence >= self.hybrid_rule_threshold:
            return RoutingDecision(
                rule.target,
                self.mode,
                f"high-confidence rule: {rule.reason}",
                rule.confidence,
                (*rule.cascade, "hybrid:accepted_rule"),
            )

        if provider is None:
            return self._fallback_to_rule(rule, "model_provider_missing")
        try:
            model = await asyncio.wait_for(
                self._model_decision(task, provider), timeout=self.model_timeout_seconds
            )
        except (TimeoutError, ValueError, json.JSONDecodeError) as exc:
            return self._fallback_to_rule(rule, f"model_failed:{type(exc).__name__}")
        return RoutingDecision(
            model.target,
            self.mode,
            model.reason,
            model.confidence,
            (*rule.cascade, *model.cascade),
        )

    def _rule_decision(self, task: PublicTaskSpec) -> RoutingDecision:
        statement = task.problem_statement.casefold()
        matched = sorted(term for term in self._ESCALATION_TERMS if term in statement)
        if matched:
            return RoutingDecision(
                RouteTarget.ESCALATED,
                self.mode,
                f"complexity terms: {', '.join(matched)}",
                0.95,
                ("rule:complexity_term",),
            )
        if task.max_changed_files > 8:
            return RoutingDecision(
                RouteTarget.ESCALATED,
                self.mode,
                f"max_changed_files={task.max_changed_files}",
                0.9,
                ("rule:wide_change_budget",),
            )
        simple_terms = ("fix", "bug", "typo", "修复", "错误")
        if any(term in statement for term in simple_terms) and task.max_changed_files <= 3:
            return RoutingDecision(
                RouteTarget.STANDARD,
                self.mode,
                "bounded bug-fix rule",
                0.85,
                ("rule:bounded_bugfix",),
            )
        return RoutingDecision(
            RouteTarget.STANDARD,
            self.mode,
            "no escalation rule matched",
            0.55,
            ("rule:default_standard",),
        )

    async def _model_decision(
        self, task: PublicTaskSpec, provider: ModelProvider
    ) -> RoutingDecision:
        request = ModelRequest(
            messages=(
                Message(
                    "system",
                    "Route the public task. Return JSON only with target "
                    '("standard" or "escalated"), reason, and confidence in [0,1]. '
                    "Do not call tools.",
                ),
                Message(
                    "user",
                    json.dumps(
                        {
                            "problem_statement": task.problem_statement,
                            "language": task.language,
                            "allowed_paths": task.allowed_paths,
                            "max_changed_files": task.max_changed_files,
                            "network_policy": task.network_policy,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ),
            tools=(),
            max_output_tokens=256,
        )
        response = await provider.complete(request)
        if response.tool_calls:
            raise ValueError("routing_model_attempted_tool_call")
        payload = json.loads(response.content)
        if not isinstance(payload, dict):
            raise ValueError("routing_response_not_object")
        target = RouteTarget(str(payload.get("target", "")))
        reason = str(payload.get("reason", "")).strip()
        confidence = float(payload.get("confidence", -1))
        if not reason:
            raise ValueError("routing_reason_missing")
        if not 0 <= confidence <= 1:
            raise ValueError("routing_confidence_out_of_range")
        return RoutingDecision(
            target,
            self.mode,
            reason,
            confidence,
            (f"model:{target.value}",),
        )

    def _fallback_to_rule(self, rule: RoutingDecision, failure: str) -> RoutingDecision:
        return RoutingDecision(
            rule.target,
            self.mode,
            f"{failure}; fallback to {rule.reason}",
            rule.confidence,
            (*rule.cascade, failure, f"fallback:{rule.target.value}"),
        )
