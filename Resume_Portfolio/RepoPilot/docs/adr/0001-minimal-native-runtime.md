# ADR 0001: Minimal native runtime before a large Agent framework

## Decision

Implement the model protocol, state machine, loop, tool registry, checkpoint, policy, trace, and
verification boundary inside RepoPilot. Do not depend on LangGraph, CrewAI, or another large
Agent framework for Stage 1–6.

## Consequences

The learning surface and failure semantics stay visible and testable. RepoPilot must maintain
its own small contracts, but can add a framework comparison later without changing Task Specs.
