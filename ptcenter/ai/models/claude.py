"""Anthropic Claude model implementation."""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseAIModel

logger = logging.getLogger(__name__)


class ClaudeModel(BaseAIModel):
    name = "claude"

    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-latest") -> None:
        self.api_key = api_key
        self.model_name = model
        self.display_name = f"Anthropic Claude ({model})"
        self.client = None
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info("Claude client initialised")
        except Exception as exc:
            logger.error("Claude init failed: %s", exc)

    def generate(self, prompt: str, system_instruction: str = "") -> Optional[str]:
        if not self.client:
            return None
        try:
            kwargs: dict = {
                "model": self.model_name,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_instruction:
                kwargs["system"] = system_instruction
            resp = self.client.messages.create(**kwargs)
            return resp.content[0].text
        except Exception as exc:
            logger.error("Claude generate error: %s", exc)
            return None

    def is_available(self) -> bool:
        return self.client is not None
