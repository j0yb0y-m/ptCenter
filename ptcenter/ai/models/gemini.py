"""Google Gemini AI model implementation."""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseAIModel

logger = logging.getLogger(__name__)


class GeminiModel(BaseAIModel):
    name = "gemini"
    display_name = "Google Gemini (gemini-2.0-flash)"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = None
        self.types = None
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialised")
        except Exception as exc:
            logger.error("Gemini init failed: %s", exc)

    def generate(self, prompt: str, system_instruction: str = "") -> Optional[str]:
        if not self.client:
            return None
        try:
            cfg = self.types.GenerateContentConfig(
                system_instruction=system_instruction or None,
                temperature=0.3,
            )
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=cfg,
            )
            return response.text
        except Exception as exc:
            logger.error("Gemini generate error: %s", exc)
            return None

    def is_available(self) -> bool:
        return self.client is not None
