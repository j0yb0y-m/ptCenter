"""Base class for all AI model implementations."""

from __future__ import annotations

from typing import Optional


class BaseAIModel:
    name: str = "base"
    display_name: str = "Base Model"

    def generate(self, prompt: str, system_instruction: str = "") -> Optional[str]:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError
