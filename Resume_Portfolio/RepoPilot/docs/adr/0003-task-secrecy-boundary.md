# ADR 0003: Separate public and evaluator task contracts

## Decision

PublicTaskSpec cannot contain hidden tests. EvaluatorTaskSpec owns hidden test identifiers and
must be loaded by a separate grader outside model context and the agent workspace.

## Consequences

Hidden evaluation is enforced by data flow and mounts, not by a prompt request. Local committed
verification scripts remain development checks and are never labelled hidden.
