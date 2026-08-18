from module import slugify

assert slugify("Hello World") == "hello-world"
assert slugify("  Agent Runtime  ") == "agent-runtime"
