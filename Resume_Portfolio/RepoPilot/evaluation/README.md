# Python Micro Benchmark v1

The ten trusted fixtures are deliberately tiny. They test the Agent Runtime, localization,
structured edits, deterministic verification, trace collection, and ablation harness—not
general software-engineering ability. Every local-model run must use a fresh copy of the case.

The committed `verify.py` files are development checks. Evaluator-only hidden tests must be
stored outside the agent workspace and mounted only into the separate grader environment.
