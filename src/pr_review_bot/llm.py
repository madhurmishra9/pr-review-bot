"""Ollama client with deterministic decoding and schema-constrained output.

Every call runs with temperature 0, top_k 1, and a fixed seed, so the same
input always produces the same output. Ollama's structured-output support
(``format`` = JSON schema) constrains generation to valid JSON matching the
schema, which removes free-form parsing failures.
"""

from __future__ import annotations

import json

import requests

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_URL = "http://localhost:11434"

# Deterministic decoding: greedy sampling plus a fixed seed. With these
# options an identical prompt yields an identical completion on every run.
DETERMINISTIC_OPTIONS = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 42,
    "repeat_penalty": 1.0,
}


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                 num_ctx: int = 16384, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.timeout = timeout

    def check_available(self) -> None:
        """Fail fast with a clear message if Ollama or the model is missing."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.base_url}. Is it running? "
                "Install: https://ollama.com — then `ollama serve`."
            ) from exc
        names = {m["name"] for m in resp.json().get("models", [])}
        base = self.model.split(":")[0]
        if self.model not in names and not any(n.split(":")[0] == base for n in names):
            raise LLMError(
                f"Model {self.model!r} is not pulled. Run: ollama pull {self.model}"
            )

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        """Single deterministic chat call returning schema-valid JSON."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,
            "options": {**DETERMINISTIC_OPTIONS, "num_ctx": self.num_ctx},
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        content = resp.json().get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned non-JSON output: {content[:200]!r}") from exc


# JSON schema the review pass must conform to.
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer"},
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low", "info"],
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "bug", "security", "performance", "correctness",
                            "error-handling", "maintainability", "style", "testing",
                        ],
                    },
                    "summary": {"type": "string"},
                    "detail": {"type": "string"},
                    "evidence": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": [
                    "line", "severity", "category", "summary", "detail", "evidence",
                ],
            },
        },
    },
    "required": ["findings"],
}

# JSON schema the verification pass must conform to.
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["confirmed", "rejected"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}

# JSON schema for the PR summary pass.
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}
