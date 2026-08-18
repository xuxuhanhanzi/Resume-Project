# ADR 0004: Fail closed for untrusted execution

## Decision

LocalTrustedRunner requires an explicit trusted fixture flag. Every untrusted test/setup command
requires DockerSandboxRunner. Missing Docker, image, or policy configuration is an error rather
than a local-process fallback.

## Consequences

Development remains easy for owned fixtures while security claims stay bounded. Docker is only
one isolation layer; hostile workloads require a pinned image and later gVisor/micro-VM review.
