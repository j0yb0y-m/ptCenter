"""Ollama local AI model implementation."""

from __future__ import annotations

import logging
from typing import Optional

import requests

from .base import BaseAIModel

logger = logging.getLogger(__name__)


class OllamaModel(BaseAIModel):
    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3") -> None:
        self.host = host.rstrip("/")
        self.model_name = model
        self.display_name = f"Ollama Local ({model})"
        self._available = self._check_connection()

    def _check_connection(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_instruction: str = "") -> Optional[str]:
        if not self._available:
            return None
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_instruction,
                "stream": False,
                "options": {"temperature": 0.3},
            }
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response")
        except Exception as exc:
            logger.error("Ollama generate error: %s", exc)
            return None

    def is_available(self) -> bool:
        return self._available
