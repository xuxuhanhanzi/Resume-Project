"""Deterministic controlled chart tasks with visual-evidence ground truth."""

from forgemm.controlled.protocol import (
    ControlledRecord,
    ControlledVerdict,
    VisualEvidence,
    parse_controlled_prediction,
    render_controlled_completion,
    verify_controlled_completion,
)

__all__ = [
    "ControlledRecord",
    "ControlledVerdict",
    "VisualEvidence",
    "parse_controlled_prediction",
    "render_controlled_completion",
    "verify_controlled_completion",
]
