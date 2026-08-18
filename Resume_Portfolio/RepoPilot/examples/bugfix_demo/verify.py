"""Immutable trusted-fixture verification command."""

from calculator import subtract

assert subtract(2, 1) == 1
assert subtract(-1, 2) == -3
print("verification passed")
