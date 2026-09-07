"""R3 typed tool-contract validation and stage-scoped exposure.

This is deliberately a pre-execution boundary.  It never grants a permission,
runs a tool, or replaces the existing policy engine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from repopilot.core.contracts import JSONValue, ToolCall, ToolResult, ToolSpec


class RecoveryCode(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    STAGE_NOT_ALLOWED = "stage_not_allowed"
    SCHEMA_INVALID = "schema_invalid"
    PATH_OUT_OF_SCOPE = "path_out_of_scope"
    PRECONDITION_FAILED = "precondition_failed"


@dataclass(frozen=True, slots=True)
class ContractDecision:
    """Structured recoverable outcome recorded by the R3 evaluator."""

    allowed: bool
    code: RecoveryCode | None = None
    detail: str = ""

    @property
    def recoverable(self) -> bool:
        return not self.allowed and self.code is not RecoveryCode.UNKNOWN_TOOL


Precondition = Callable[[ToolCall], str | None]


class StageScopedToolContracts:
    """Validate tool name, phase, JSON shape, path scope, then preconditions."""

    def __init__(
        self,
        specs: tuple[ToolSpec, ...],
        *,
        allowlist: Mapping[str, frozenset[str]],
        preconditions: Mapping[str, Precondition] | None = None,
    ) -> None:
        self.specs = {spec.name: spec for spec in specs}
        if len(self.specs) != len(specs):
            raise ValueError("tool contract specs must have unique names")
        if not allowlist or any(not stage.strip() for stage in allowlist):
            raise ValueError("stage allowlist must not be empty")
        if any(not tools <= self.specs.keys() for tools in allowlist.values()):
            raise ValueError("stage allowlist refers to an unknown tool")
        self.allowlist = {stage: frozenset(tools) for stage, tools in allowlist.items()}
        self.preconditions = dict(preconditions or {})
        if not self.preconditions.keys() <= self.specs.keys():
            raise ValueError("precondition refers to an unknown tool")

    def visible_specs(self, stage: str) -> tuple[ToolSpec, ...]:
        """Return deterministic model-visible schemas without executing a tool."""

        allowed = self._allowed_for_stage(stage)
        return tuple(self.specs[name] for name in sorted(allowed))

    def validate(self, call: ToolCall, *, stage: str) -> ContractDecision:
        spec = self.specs.get(call.name)
        if spec is None:
            return ContractDecision(False, RecoveryCode.UNKNOWN_TOOL, "tool is not registered")
        if call.name not in self._allowed_for_stage(stage):
            return ContractDecision(
                False,
                RecoveryCode.STAGE_NOT_ALLOWED,
                f"{call.name} is not exposed during {stage}",
            )
        error = validate_json_schema(call.arguments, spec.input_schema)
        if error is not None:
            return ContractDecision(False, RecoveryCode.SCHEMA_INVALID, error)
        path_error = _validate_declared_paths(call.arguments, spec.input_schema)
        if path_error is not None:
            return ContractDecision(False, RecoveryCode.PATH_OUT_OF_SCOPE, path_error)
        predicate = self.preconditions.get(call.name)
        if predicate is not None:
            detail = predicate(call)
            if detail is not None:
                return ContractDecision(False, RecoveryCode.PRECONDITION_FAILED, detail)
        return ContractDecision(True)

    def next_stage(self, stage: str, call: ToolCall, result: ToolResult) -> str:
        """Advance by observed success only; model text never changes a stage."""

        self._allowed_for_stage(stage)
        if not result.ok:
            return "modify" if stage == "verify" and call.name == "run_tests" else stage
        if stage == "explore" and call.name in {
            "list_files",
            "read_file",
            "search_text",
            "find_symbol",
            "retrieve_code",
        }:
            return "modify" if "modify" in self.allowlist else stage
        if stage == "modify" and call.name == "apply_patch":
            return "verify" if "verify" in self.allowlist else stage
        if stage == "verify" and call.name == "run_tests":
            return "completed" if "completed" in self.allowlist else stage
        return stage

    def _allowed_for_stage(self, stage: str) -> frozenset[str]:
        try:
            return self.allowlist[stage]
        except KeyError as error:
            raise ValueError(f"unknown tool stage: {stage}") from error


def validate_json_schema(
    value: object, schema: Mapping[str, JSONValue], *, path: str = "$"
) -> str | None:
    """Validate the small, deterministic JSON-Schema subset used by ToolSpec."""

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return "tool schema is invalid"
        if any(name not in value for name in required):
            missing = next(name for name in required if name not in value)
            return f"{path}.{missing} is required"
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(properties))
            if unexpected:
                return f"{path}.{unexpected[0]} is not allowed"
        for name, child_schema in properties.items():
            if name in value:
                if not isinstance(child_schema, dict):
                    return f"tool schema for {name} is invalid"
                error = validate_json_schema(value[name], child_schema, path=f"{path}.{name}")
                if error is not None:
                    return error
        return None
    if schema_type == "string":
        if not isinstance(value, str):
            return f"{path} must be a string"
        if "minLength" in schema and len(value) < schema["minLength"]:
            return f"{path} is shorter than minLength"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return f"{path} exceeds maxLength"
    elif schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"{path} must be an integer"
    elif schema_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{path} must be a number"
    elif schema_type == "boolean":
        if not isinstance(value, bool):
            return f"{path} must be a boolean"
    elif schema_type == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = validate_json_schema(item, item_schema, path=f"{path}[{index}]")
                if error is not None:
                    return error
    elif schema_type is not None:
        return f"unsupported schema type {schema_type}"
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return f"{path} must be one of the declared enum values"
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        return f"{path} is below minimum"
    if "maximum" in schema and isinstance(value, (int, float)) and value > schema["maximum"]:
        return f"{path} exceeds maximum"
    return None


def _validate_declared_paths(
    arguments: dict[str, JSONValue], schema: Mapping[str, JSONValue]
) -> str | None:
    """Apply an explicit x-path-fields extension without resolving a filesystem."""

    fields = schema.get("x-path-fields", [])
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        return "tool schema has invalid x-path-fields"
    for field in fields:
        value = arguments.get(field)
        if not isinstance(value, str):
            continue
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            return f"$.{field} must remain inside the workspace scope"
    return None


def stage_scoped_coding_contracts(specs: tuple[ToolSpec, ...]) -> StageScopedToolContracts:
    """Create the R3 profile from existing coding specs without changing policy.

    The profile is opt-in.  It exposes only tools already registered by the
    caller and adds a narrow path-scope extension to file-editing schemas.
    """

    path_tools = {"read_file", "apply_patch"}
    scoped_specs: list[ToolSpec] = []
    for spec in specs:
        schema = dict(spec.input_schema)
        if spec.name in path_tools:
            schema["x-path-fields"] = ["path"]
        scoped_specs.append(
            ToolSpec(
                spec.name,
                spec.description,
                schema,
                spec.permission,
                spec.timeout_seconds,
                spec.read_only,
                spec.idempotent,
            )
        )
    available = {spec.name for spec in scoped_specs}
    desired = {
        "explore": {"list_files", "read_file", "search_text", "find_symbol", "retrieve_code"},
        "modify": {
            "list_files",
            "read_file",
            "search_text",
            "find_symbol",
            "retrieve_code",
            "apply_patch",
        },
        "verify": {"run_tests", "git_diff", "read_file"},
        "completed": set(),
    }
    return StageScopedToolContracts(
        tuple(scoped_specs),
        allowlist={stage: frozenset(names & available) for stage, names in desired.items()},
    )
