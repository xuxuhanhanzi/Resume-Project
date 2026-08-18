"""Minimal OpenAI-compatible server serving a HuggingFace model via transformers.

Used for R4-A (1.5B+adapter) evaluation when Ollama's --experimental
safetensors import fails on Windows (MLX runner unavailable).

Listens on http://127.0.0.1:8080/v1/chat/completions and returns standard
OpenAI chat-completion JSON.  Single-worker, greedy-decoding by default.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

_INFERENCE_LOCK = threading.Lock()


def _load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, messages: list, temperature: float, max_tokens: int) -> str:
    import torch

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad(), _INFERENCE_LOCK:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 0.01) if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def make_handler(model, tokenizer):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                self.send_error(404, "not found")
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self.send_error(400, "invalid JSON")
                return
            messages = payload.get("messages", [])
            temperature = float(payload.get("temperature", 0))
            max_tokens = int(payload.get("max_tokens", 1024))
            try:
                content = _generate(model, tokenizer, messages, temperature, max_tokens)
            except Exception as exc:  # noqa: BLE001
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode())
                return
            response = {
                "id": f"chatcmpl-{uuid4().hex[:12]}",
                "object": "chat.completion",
                "model": payload.get("model", "hf-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/v1/models":
                data = json.dumps({"data": [{"id": "hf-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        def log_message(self, format, *args):  # noqa: A002, N802
            pass  # silence default logging

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    print(f"[serve] loading model from {args.model_path}", flush=True)
    model, tokenizer = _load_model(args.model_path)
    print(f"[serve] model loaded, starting server on port {args.port}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, tokenizer))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
