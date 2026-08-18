from module import clamp

assert clamp(5, 0, 10) == 5
assert clamp(-2, 0, 10) == 0
assert clamp(20, 0, 10) == 10
