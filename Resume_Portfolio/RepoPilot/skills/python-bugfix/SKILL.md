---
name: python-bugfix
description: Safely diagnose and repair a Python bug with deterministic verification.
---

# Python Bug Fix Procedure

1. Inspect the task contract and repository structure before editing.
2. Search for the failing behavior by symbol and exact text; read only relevant ranges.
3. Form a small falsifiable hypothesis and prefer the narrowest correct patch.
4. Apply exact replacements through `apply_patch`; never weaken or delete tests.
5. Run the immutable task test command and inspect failures before retrying.
6. Review the final diff for unrelated changes, then propose finish. The verifier decides success.
