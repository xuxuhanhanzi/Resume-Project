"""Public task, private evaluator, environment, and output boundaries."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol

from repopilot.core.budgets import RunBudget
from repopilot.core.contracts import JSONValue

_FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "gold_answer",
    "gold_patch",
    "gold_patch_sha256",
    "hidden_tests",
    "oracle",
    "oracle_ref",
}


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"mount_path must be relative and traversal-free: {value!r}")


def _reject_private_metadata(value: JSONValue, *, location: str = "public_metadata") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"{location} contains evaluator-only key: {key}")
            _reject_private_metadata(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_private_metadata(child, location=f"{location}[{index}]")


class BenchmarkDomain(StrEnum):
    """The three initially registered evaluation domains."""

    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    SOFTWARE_ENGINEERING = "software_engineering"


class NetworkPolicy(StrEnum):
    """Network boundary applied by a task environment."""

    DENY = "deny"
    ALLOWLIST = "allowlist"


@dataclass(frozen=True, slots=True)
class AssetRef:
    """A public, content-addressable input made available to an agent."""

    uri: str
    sha256: str | None = None
    mount_path: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.uri, "asset uri")
        if self.sha256 is not None:
            _validate_sha256(self.sha256)
        if self.mount_path is not None:
            _validate_relative_path(self.mount_path)


@dataclass(frozen=True, slots=True)
class SecretRef:
    """Opaque evaluator-store reference that is never serialized with a public task."""

    store_id: str
    key: str

    def __post_init__(self) -> None:
        _require_text(self.store_id, "secret store_id")
        _require_text(self.key, "secret key")


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """Domain-neutral information that is safe to expose to an agent."""

    benchmark_id: str
    dataset_revision: str
    task_id: str
    domain: BenchmarkDomain
    instruction: str
    environment_id: str
    assets: tuple[AssetRef, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    budget: RunBudget = field(default_factory=RunBudget)
    network_policy: NetworkPolicy = NetworkPolicy.DENY
    public_metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "benchmark_id",
            "dataset_revision",
            "task_id",
            "instruction",
            "environment_id",
        ):
            _require_text(str(getattr(self, field_name)), field_name)
        if len(set(self.allowed_tools)) != len(self.allowed_tools):
            raise ValueError("allowed_tools must not contain duplicates")
        if any(not tool.strip() for tool in self.allowed_tools):
            raise ValueError("allowed_tools must be non-empty names")
        _reject_private_metadata(self.public_metadata)


@dataclass(frozen=True, slots=True)
class EvaluatorCase:
    """Evaluator-only configuration, kept outside model context and sandbox mounts."""

    benchmark_id: str
    dataset_revision: str
    task_id: str
    evaluator_id: str
    oracle_ref: SecretRef
    evaluator_config: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("benchmark_id", "dataset_revision", "task_id", "evaluator_id"):
            _require_text(str(getattr(self, field_name)), field_name)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A durable output produced by a task run."""

    path: str
    sha256: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.path, "artifact path")
        if self.sha256 is not None:
            _validate_sha256(self.sha256)


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A source location cited by the task output."""

    source_id: str
    locator: str

    def __post_init__(self) -> None:
        _require_text(self.source_id, "evidence source_id")
        _require_text(self.locator, "evidence locator")


@dataclass(frozen=True, slots=True)
class TaskOutput:
    """Provider-neutral result returned by an executor."""

    final_answer: str | None = None
    structured_payload: Mapping[str, JSONValue] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    changed_files: tuple[str, ...] = ()
    run_metrics: SharedRunMetrics = field(default_factory=lambda: SharedRunMetrics())


@dataclass(frozen=True, slots=True)
class SharedRunMetrics:
    """Comparable engineering metrics shared by all benchmark domains."""

    iterations: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    recovery_count: int = 0
    invalid_tool_calls: int = 0
    budget_exhausted: bool = False
    security_blocks: int = 0

    def __post_init__(self) -> None:
        integer_values = (
            self.iterations,
            self.tool_calls,
            self.input_tokens,
            self.output_tokens,
            self.recovery_count,
            self.invalid_tool_calls,
            self.security_blocks,
        )
        if any(value < 0 for value in integer_values) or self.wall_seconds < 0:
            raise ValueError("shared run metrics must be non-negative")


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Normalized result with one domain-specific primary metric."""

    task_success: bool
    primary_metric_name: str
    primary_metric_value: float
    domain_metrics: Mapping[str, float] = field(default_factory=dict)
    shared_metrics: SharedRunMetrics = field(default_factory=SharedRunMetrics)
    failure_type: str | None = None
    evaluator_trace_ref: str = ""

    def __post_init__(self) -> None:
        _require_text(self.primary_metric_name, "primary_metric_name")
        numeric_values = (self.primary_metric_value, *self.domain_metrics.values())
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("evaluation metrics must be finite")


@dataclass(frozen=True, slots=True)
class PreparedTask:
    """Public execution material produced by an adapter."""

    task: BenchmarkTask
    work_dir: Path
    public_payload: object | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentHandle:
    """An active task environment visible to the executor and evaluator."""

    handle_id: str
    work_dir: Path
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.handle_id, "environment handle_id")


class BenchmarkAdapter(Protocol):
    """Convert one frozen dataset into public tasks and private evaluator cases."""

    @property
    def benchmark_id(self) -> str: ...

    def load(
        self, split: str, task_ids: Sequence[str] | None = None
    ) -> Iterable[BenchmarkTask]: ...

    def prepare(self, task: BenchmarkTask, run_dir: Path) -> PreparedTask: ...

    def evaluator_case(self, task_id: str) -> EvaluatorCase: ...


class TaskEnvironment(Protocol):
    """Lifecycle boundary for local, container, or repository environments."""

    async def start(self, prepared: PreparedTask) -> EnvironmentHandle: ...

    async def reset(self, handle: EnvironmentHandle) -> None: ...

    async def close(self, handle: EnvironmentHandle) -> None: ...


class TaskExecutor(Protocol):
    """Run an agent without receiving evaluator-only data."""

    async def execute(
        self, task: BenchmarkTask, prepared: PreparedTask, handle: EnvironmentHandle
    ) -> TaskOutput: ...


class TaskEvaluator(Protocol):
    """Score an output after agent execution has finished."""

    async def evaluate(
        self, output: TaskOutput, case: EvaluatorCase, handle: EnvironmentHandle
    ) -> EvaluationOutcome: ...
