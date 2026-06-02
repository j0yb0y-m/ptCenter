"""
ptcenter.ai.manager
===================
Manages multiple AI models, tracks conversation history, and provides
chunked analysis for large scan outputs.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .models.base import BaseAIModel
from .models.gemini import GeminiModel
from .models.openai import OpenAIModel
from .models.claude import ClaudeModel
from .models.ollama import OllamaModel

console = Console()
logger = logging.getLogger(__name__)

_PENTEST_SYSTEM = (
    "You are an expert penetration tester and CTF player. "
    "Answer questions about offensive security, vulnerabilities, tools, and techniques. "
    "Be concise, technical, and practical."
)

_ANALYSIS_SYSTEM = (
    "You are an expert penetration tester and security analyst. "
    "Provide clear, actionable security analysis. Be concise but thorough. "
    "Highlight critical issues and provide practical remediation steps."
)

_CHUNK_SIZE = 7500


class AIManager:
    """Manages multiple AI models and tracks conversation history."""

    def __init__(self) -> None:
        self.models: dict[str, BaseAIModel] = {}
        self.active_model: Optional[BaseAIModel] = None
        self.conversation_history: list[dict] = []
        self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            m = GeminiModel(gemini_key)
            if m.is_available():
                self.models["gemini"] = m

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            m = OpenAIModel(openai_key, model=os.getenv("OPENAI_MODEL", "gpt-4o"))
            if m.is_available():
                self.models["openai"] = m

        claude_key = os.getenv("ANTHROPIC_API_KEY")
        if claude_key:
            m = ClaudeModel(claude_key, model=os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-latest"))
            if m.is_available():
                self.models["claude"] = m

        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        m = OllamaModel(host=ollama_host, model=ollama_model)
        if m.is_available():
            self.models["ollama"] = m

        saved = os.getenv("ACTIVE_AI_MODEL")
        if saved and saved in self.models:
            self.active_model = self.models[saved]
        elif self.models:
            self.active_model = next(iter(self.models.values()))

        if self.active_model:
            logger.info("Active AI: %s", self.active_model.display_name)
        else:
            logger.warning("No AI models available")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self.active_model is not None and self.active_model.is_available()

    def active_model_name(self) -> str:
        if self.active_model:
            return self.active_model.display_name
        return "None (no model configured)"

    def get_available_models(self) -> dict[str, BaseAIModel]:
        return self.models

    def select_model(self, key: str) -> bool:
        if key in self.models:
            self.active_model = self.models[key]
            return True
        return False

    def generate(self, prompt: str, system_instruction: str = "") -> Optional[str]:
        if not self.active_model:
            return None
        return self.active_model.generate(prompt, system_instruction)

    # ------------------------------------------------------------------
    # Chunked analysis
    # ------------------------------------------------------------------

    def analyze_scan(self, scan_result: str, scan_type: str) -> Optional[str]:
        """
        Analyse *scan_result* with the active model.
        If output > _CHUNK_SIZE chars, split it into chunks, summarise
        each individually, then synthesise a final analysis.
        """
        if not self.is_available():
            console.print("[yellow]⚠[/] AI unavailable — configure an API key")
            return None

        console.print(f"[bold blue]🤖[/] Analysing with {self.active_model_name()}…")

        prompt_template = (
            "Analyse this {scan_type} scan result and provide:\n"
            "1. Executive Summary (2-3 sentences)\n"
            "2. Identified vulnerabilities or security issues\n"
            "3. Risk Assessment (Critical/High/Medium/Low for each finding)\n"
            "4. Recommended next steps and mitigation strategies\n"
            "5. Additional reconnaissance suggestions\n\n"
            "Scan Results:\n{results}"
        )

        if len(scan_result) < _CHUNK_SIZE:
            prompt = prompt_template.format(scan_type=scan_type, results=scan_result)
            return self.active_model.generate(prompt, _ANALYSIS_SYSTEM)

        # Chunked path
        chunks = [scan_result[i:i+_CHUNK_SIZE] for i in range(0, len(scan_result), _CHUNK_SIZE)]
        summaries: list[str] = []
        for idx, chunk in enumerate(chunks, 1):
            console.print(f"[dim]  Summarising chunk {idx}/{len(chunks)}…[/]")
            chunk_prompt = (
                f"Summarise the security-relevant content from chunk {idx} of "
                f"a {scan_type} scan in 5-8 bullet points:\n\n{chunk}"
            )
            summary = self.active_model.generate(chunk_prompt, _ANALYSIS_SYSTEM)
            if summary:
                summaries.append(f"--- Part {idx} ---\n{summary}")

        combined = "\n\n".join(summaries)
        final_prompt = (
            f"Based on these per-chunk summaries of a full {scan_type} scan, "
            "provide a unified security analysis:\n"
            "1. Executive Summary\n"
            "2. All vulnerabilities found\n"
            "3. Risk levels\n"
            "4. Remediation priorities\n\n"
            f"{combined}"
        )
        return self.active_model.generate(final_prompt, _ANALYSIS_SYSTEM)

    # ------------------------------------------------------------------
    # Agentic recon
    # ------------------------------------------------------------------

    def decide_next_action(self, scan_output: str, context: str = "") -> Optional[dict]:
        """
        Given scan output, ask the AI to decide the next best recon action.
        Returns a validated JSON dict or None.
        """
        if not self.is_available():
            return None

        prompt = (
            "You are an automated pentest agent. Given these scan results, "
            "decide the single best next action. "
            'Respond ONLY with valid JSON — no markdown, no extra text:\n'
            '{"tool": "nmap|nikto|gobuster|sqlmap|sslscan|enum4linux", '
            '"flags": "...", "target": "...", "reason": "..."}\n\n'
            f"Context: {context}\n\nScan Results:\n{scan_output[:6000]}"
        )
        raw = self.active_model.generate(prompt, _ANALYSIS_SYSTEM)
        if not raw:
            return None
        try:
            # Strip any accidental markdown fences
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interactive chat
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> Optional[str]:
        """
        Send *user_message* in a conversation context (history-aware).
        History is capped at the last 20 turns to avoid token overflow.
        """
        if not self.is_available():
            return None

        self.conversation_history.append({"role": "user", "content": user_message})
        if len(self.conversation_history) > 40:  # 20 turns * 2 roles
            self.conversation_history = self.conversation_history[-40:]

        # Build a single prompt with history prefix for models that don't
        # natively support message arrays (Gemini, Ollama)
        history_text = ""
        for turn in self.conversation_history[:-1]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n\n"

        prompt = history_text + f"User: {user_message}"
        response = self.active_model.generate(prompt, _PENTEST_SYSTEM)

        if response:
            self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self) -> None:
        self.conversation_history.clear()

    # ------------------------------------------------------------------
    # Display helper
    # ------------------------------------------------------------------

    def display_analysis(self, analysis: str, title: str = "AI Security Analysis") -> None:
        panel = Panel(
            Markdown(analysis),
            title=f"[bold cyan]🤖 {title}[/]",
            border_style="cyan",
            padding=(1, 2),
        )
        console.print(panel)
