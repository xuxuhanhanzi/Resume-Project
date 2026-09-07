from __future__ import annotations

from repopilot.evaluation.bfcl import BFCLInstance, _contract_decision, _normalise_schema
from repopilot.tools.contracts import RecoveryCode


def test_bfcl_aliases_normalise_to_the_contract_schema_subset() -> None:
    schema = _normalise_schema(
        {
            "type": "dict",
            "properties": {
                "amount": {"type": "float"},
                "names": {"type": "list", "items": {"type": "string"}},
            },
            "required": ["amount"],
        }
    )

    assert schema["type"] == "object"
    assert schema["properties"]["amount"]["type"] == "number"
    assert schema["properties"]["names"]["type"] == "array"
    assert schema["additionalProperties"] is False


def test_bfcl_contract_decision_surfaces_unknown_tool_without_execution() -> None:
    item = BFCLInstance(
        instance_id="simple_python_0",
        category="simple_python",
        question="Calculate a triangle area.",
        functions=(
            {
                "name": "calculate_triangle_area",
                "parameters": {
                    "type": "dict",
                    "properties": {"base": {"type": "integer"}},
                    "required": ["base"],
                },
            },
        ),
    )

    good = _contract_decision(
        item,
        [{"name": "calculate_triangle_area", "arguments": {"base": 10}}],
    )
    unknown = _contract_decision(item, [{"name": "wrong_tool", "arguments": {}}])

    assert good is not None and good.allowed
    assert unknown is not None and unknown.code is RecoveryCode.UNKNOWN_TOOL
