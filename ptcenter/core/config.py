"""
ptcenter.core.config
====================
Load / save ~/.ptcenter_config.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILE = Path.home() / ".ptcenter_config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "output_directory": str(Path("/tmp/ptcenter_outputs")),
    "timeout": 300,
    "auto_ai_analysis": True,
    "save_logs": True,
    "tester_name": "Unknown",
}


def load_config() -> dict[str, Any]:
    """Return merged config (defaults + saved overrides)."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as fh:
                loaded = json.load(fh)
            return {**DEFAULT_CONFIG, **loaded}
        except Exception as exc:
            logger.warning("Could not load config: %s — using defaults", exc)
    return dict(DEFAULT_CONFIG)


def save_config(data: dict[str, Any]) -> None:
    """Persist *data* to the config file."""
    try:
        with open(CONFIG_FILE, "w") as fh:
            json.dump(data, fh, indent=4)
    except Exception as exc:
        logger.warning("Could not save config: %s", exc)
