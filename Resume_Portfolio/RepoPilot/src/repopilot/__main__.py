"""Allow ``python -m repopilot`` to invoke the CLI."""

from repopilot.cli import entrypoint

if __name__ == "__main__":
    raise SystemExit(entrypoint())
