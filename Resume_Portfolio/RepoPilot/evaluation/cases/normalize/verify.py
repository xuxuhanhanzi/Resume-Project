from module import normalize

assert normalize("  HeLLo ") == "hello"
assert normalize("AGENT") == "agent"
