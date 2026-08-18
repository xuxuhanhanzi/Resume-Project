from module import safe_get

assert safe_get({"a": 1}, "a", 0) == 1
assert safe_get({}, "missing", 7) == 7
