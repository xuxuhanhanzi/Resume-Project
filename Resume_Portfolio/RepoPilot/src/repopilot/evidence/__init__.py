"""Immutable evaluation identities, receipts, and small-sample statistics."""

from repopilot.evidence.protocol import (
    DatasetInstance,
    ExperimentReceipt,
    FrozenManifest,
    build_grouped_split_manifest,
    sha256_file,
)
from repopilot.evidence.statistics import (
    BootstrapInterval,
    WilsonInterval,
    paired_bootstrap_mean_difference,
    wilson_interval,
)

__all__ = [
    "BootstrapInterval",
    "DatasetInstance",
    "ExperimentReceipt",
    "FrozenManifest",
    "WilsonInterval",
    "build_grouped_split_manifest",
    "paired_bootstrap_mean_difference",
    "sha256_file",
    "wilson_interval",
]
