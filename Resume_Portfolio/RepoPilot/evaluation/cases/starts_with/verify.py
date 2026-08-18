from module import starts_with

assert starts_with("agent-runtime", "agent")
assert not starts_with("agent-runtime", "runtime")
