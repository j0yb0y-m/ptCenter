"""OpenAI GPT model implementation."""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseAIModel

logger = logging.getLogger(__name__)


class OpenAIModel(BaseAIModel):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self.api_key = api_key
        self.model_name = model
        self.display_name = f"OpenAI {model}"
        self.client = None
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialised")
        except Exception as exc:
            logger.error("OpenAI init failed: %s", exc)

    def generate(self, prompt: str, system_instruction: str = "") -> Optional[str]:
        if not self.client:
            return None
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            logger.error("OpenAI generate error: %s", exc)
            return None

    def is_available(self) -> bool:
        return self.client is not None
