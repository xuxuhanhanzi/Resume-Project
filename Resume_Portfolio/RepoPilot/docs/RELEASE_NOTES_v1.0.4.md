# RepoPilot v1.0.4

本地实验闭环版本。固定 `qwen2.5:7b` digest，通过 RepoPilot OpenAI-compatible provider
完成 1/4/8 并发各 24 个真实模型请求。72/72 非空；结果作为本机短请求模型服务容量，
与 ScriptedProvider 的端到端 Runtime/Verifier 调度容量严格分开。
