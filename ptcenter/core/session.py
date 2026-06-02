"""
ptcenter.core.session
=====================
Session save / load.  One session = one engagement or CTF box.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

console = Console()

SESSIONS_DIR = Path.home() / ".ptcenter_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# Data helpers

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"

# Public API
def new_session(name: str, target: str, scope: list[str]) -> dict[str, Any]:
    """Create and immediately persist a new session dict."""
    session: dict[str, Any] = {
        "session_id": str(uuid.uuid4()),
        "name": name,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "target": target,
        "scope": scope,
        "notes": "",
        "findings": [],
        "scans_completed": [],
    }
    save_session(session)
    return session


def save_session(session: dict[str, Any]) -> None:
    """Persist *session* to disk."""
    session["updated_at"] = _now_iso()
    path = _session_path(session["session_id"])
    with open(path, "w") as fh:
        json.dump(session, fh, indent=2)


def load_session(session_id: str) -> Optional[dict[str, Any]]:
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception:
        return None


def list_sessions() -> list[dict[str, Any]]:
    """Return all saved sessions (summary fields only), newest first."""
    sessions = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            sessions.append({
                "session_id": data.get("session_id", f.stem),
                "name": data.get("name", "Unnamed"),
                "target": data.get("target", "?"),
                "created_at": data.get("created_at", ""),
                "findings_count": len(data.get("findings", [])),
            })
        except Exception:
            continue
    sessions.sort(key=lambda s: s["created_at"], reverse=True)
    return sessions


def add_finding(
    session: dict[str, Any],
    module: str,
    tool: str,
    severity: str,
    title: str,
    details: str,
    output_file: str = "",
) -> None:
    """Append a finding to *session* and save immediately."""
    finding = {
        "timestamp": _now_iso(),
        "module": module,
        "tool": tool,
        "severity": severity,
        "title": title,
        "details": details,
        "output_file": output_file,
    }
    session.setdefault("findings", []).append(finding)
    if tool not in session.get("scans_completed", []):
        session.setdefault("scans_completed", []).append(tool)
    save_session(session)


def mark_scan_complete(session: dict[str, Any], scan_name: str) -> None:
    session.setdefault("scans_completed", [])
    if scan_name not in session["scans_completed"]:
        session["scans_completed"].append(scan_name)
    save_session(session)


# Display helpers
def display_session_summary(session: dict[str, Any]) -> None:
    """Print a Rich table summarising the session's findings."""
    table = Table(
        title=f"[bold cyan]Session: {session['name']}[/]  —  Target: {session['target']}",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Severity", width=10)
    table.add_column("Title")
    table.add_column("Module", width=12)
    table.add_column("Tool", width=14)
    table.add_column("Timestamp", width=22)

    severity_colors = {
        "Critical": "bold red",
        "High": "red",
        "Medium": "yellow",
        "Low": "green",
        "Info": "blue",
    }

    for idx, f in enumerate(session.get("findings", []), 1):
        sev = f.get("severity", "Info")
        color = severity_colors.get(sev, "white")
        table.add_row(
            str(idx),
            f"[{color}]{sev}[/]",
            f.get("title", ""),
            f.get("module", ""),
            f.get("tool", ""),
            f.get("timestamp", "")[:19],
        )

    if not session.get("findings"):
        table.add_row("-", "[dim]—[/]", "[dim]No findings yet[/]", "", "", "")

    console.print(table)
    console.print(f"[dim]Scans completed: {', '.join(session.get('scans_completed', [])) or 'none'}[/]")
