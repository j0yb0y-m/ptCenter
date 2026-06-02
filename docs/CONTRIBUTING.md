# ptCenter v2 — Contributing Guide

Contributions are welcome — whether it's a bug report, a new module, an additional AI provider, UI improvements, or documentation fixes.

---

## Code of Conduct

This project exists to support authorized security research and education. Contributions must not:

- Add functionality designed to facilitate illegal access to systems
- Remove or weaken input validation / shell injection protections
- Introduce hardcoded credentials or API keys
- Add dependencies that phone home without user knowledge

---

## Getting Started

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/ptcenter.git
cd ptcenter

# Set up dev environment with all extras
uv sync --extra full

# Create a feature branch
git checkout -b feature/my-new-thing
```

---

## Project Structure

```
ptcenter/
├── __main__.py          # Entry point + CLI argparse
├── core/
│   ├── app.py           # Main app class & menu routing → ADD YOUR MENU HANDLER HERE
│   ├── config.py        # Config I/O (no changes usually needed)
│   ├── reporter.py      # HTML report (add finding categories here if needed)
│   ├── runner.py        # Subprocess wrapper (do not bypass)
│   ├── session.py       # Session persistence
│   ├── sysinfo.py       # Distro detection + install hints → ADD TOOL HINTS HERE
│   └── validator.py     # Input validation → ADD NEW VALIDATORS HERE
├── ai/
│   ├── manager.py       # AI orchestration (no changes usually needed)
│   └── models/          # → ADD NEW AI PROVIDER HERE
│       ├── base.py
│       ├── gemini.py
│       ├── openai.py
│       ├── claude.py
│       └── ollama.py
├── modules/             # → ADD NEW MODULE HERE
│   ├── scanner.py
│   ├── osint.py
│   └── ...
└── ui/
    ├── banner.py
    ├── menu.py          # → ADD YOUR MENU DISPLAY FUNCTION HERE
    └── colors.py
```

---

## Adding a New Module

### Step 1 — Create the module file

Create `ptcenter/modules/mymodule.py`. Follow this template:

```python
"""
ptcenter.modules.mymodule
=========================
Brief description of what this module does.
All subprocess calls must go through core.runner — shell=False always.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel

from ptcenter.core.runner import run_command, stream_command
from ptcenter.core.sysinfo import install_hint
from ptcenter.core.validator import validate_target  # import what you need

console = Console()


def _ts() -> str:
    """Timestamp string for output filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _check(tool: str) -> bool:
    """Return True if the tool binary is on PATH."""
    return shutil.which(tool) is not None


def my_function(output_dir: Path, ai_manager: Any = None) -> None:
    """
    Short description of what this function does.

    Args:
        output_dir:  Directory to write output files.
        ai_manager:  Optional AIManager instance for analysis.
    """
    # 1 — Check required binary
    if not _check("mytool"):
        console.print(f"[red]✗[/] mytool not installed.  [dim]{install_hint('mytool')}[/]")
        return

    # 2 — Get and validate user input
    target = console.input("[bold white][+] Target: [/]").strip()
    if not target or not validate_target(target):
        console.print("[red]✗[/] Invalid target")
        return

    # 3 — Build command as a list (NEVER as a string with shell=True)
    cmd = ["mytool", "--flag", target]
    out_file = str(output_dir / f"mymodule_{_ts()}.txt")

    # 4 — Execute safely
    ok, result = run_command(cmd, output_file=out_file, timeout=120)
    if not ok:
        return

    # 5 — Optional AI analysis
    if ai_manager and ai_manager.is_available():
        analysis = ai_manager.analyze_scan(result, "mymodule")
        if analysis:
            ai_manager.display_analysis(analysis, "My Module Analysis")

    console.print(f"[bold green]✓[/] Output saved: [cyan]{out_file}[/]")
```

**Key rules:**
- Always use `run_command()` or `stream_command()` from `core.runner` — never `subprocess.run()` directly.
- Never use `shell=True` or pass raw user input as a string to a subprocess.
- Always validate input with a function from `core.validator` before using it.
- Accept `output_dir: Path` and `ai_manager: Any = None` as standard parameters.

---

### Step 2 — Add install hints for your tool

In `ptcenter/core/sysinfo.py`, add your tool to `_PKG_MAP`:

```python
_PKG_MAP: dict[str, dict[str, str]] = {
    # ... existing entries ...
    "mytool": {
        "arch":   "sudo pacman -S mytool",      # or AUR
        "debian": "sudo apt install mytool",
        "fedora": "sudo dnf install mytool",
    },
}
```

---

### Step 3 — Add a menu display function

In `ptcenter/ui/menu.py`, add a function that prints your module's sub-menu using the existing Rich style:

```python
def mymodule_menu() -> None:
    console.print(
        "\n[bold cyan]══ My Module ══[/]\n"
        "\n"
        "  [magenta]1[/]  My First Function\n"
        "  [magenta]2[/]  My Second Function\n"
        "  [magenta]3[/]  Back\n"
    )
```

---

### Step 4 — Wire it into the main app

In `ptcenter/core/app.py`:

**Import your module at the top:**
```python
from ptcenter.modules import (
    exploit,
    mymodule,   # ← add this
    network,
    ...
)
```

**Add a menu handler method to the `PTCenter` class:**
```python
def _mymodule_menu(self) -> None:
    while True:
        ui.mymodule_menu()
        choice = console.input("[bold white][★] Select: [/]").strip()
        if choice == "1":
            mymodule.my_function(self.output_dir, self.ai_manager)
        elif choice == "2":
            mymodule.my_second_function(self.output_dir)
        elif choice == "3":
            break
        else:
            console.print("[red]✗[/] Invalid option")
```

**Add it to the main menu in `_interactive()`:**
```python
# In the main menu loop:
elif choice == "14":    # use the next available number
    self._mymodule_menu()
```

**Update `ui/menu.py`'s `main_menu()` function** to show the new option.

---

### Step 5 — Update documentation

- Add your module to `docs/MODULES.md` with a full sub-tool table.
- Add any new binary dependencies to the tool table in `README.md`.

---

## Adding a New AI Provider

All AI model implementations live in `ptcenter/ai/models/`. They inherit from `BaseAIModel`.

### Step 1 — Create the model file

Create `ptcenter/ai/models/myprovider.py`:

```python
"""My Provider AI model implementation."""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseAIModel

logger = logging.getLogger(__name__)


class MyProviderModel(BaseAIModel):
    name = "myprovider"
    display_name = "My Provider (my-model-name)"

    def __init__(self, api_key: str, model: str = "my-default-model") -> None:
        self.api_key = api_key
        self.model_name = model
        self.client = None
        try:
            import myprovider_sdk
            self.client = myprovider_sdk.Client(api_key=api_key)
            logger.info("MyProvider client initialised")
        except Exception as exc:
            logger.error("MyProvider init failed: %s", exc)

    def generate(self, prompt: str, system_instruction: str = "") -> Optional[str]:
        if not self.client:
            return None
        try:
            response = self.client.chat(
                model=self.model_name,
                system=system_instruction,
                message=prompt,
                temperature=0.3,
            )
            return response.text
        except Exception as exc:
            logger.error("MyProvider generate error: %s", exc)
            return None

    def is_available(self) -> bool:
        return self.client is not None
```

**Required:**
- `name` class attribute — used as the `.env` key (`ACTIVE_AI_MODEL=myprovider`)
- `display_name` — shown in the UI
- `generate(prompt, system_instruction)` — returns `Optional[str]`
- `is_available()` — returns `bool`

---

### Step 2 — Register in AIManager

In `ptcenter/ai/manager.py`, add your model to `_load_models()`:

```python
from .models.myprovider import MyProviderModel

def _load_models(self) -> None:
    # ... existing providers ...

    myprovider_key = os.getenv("MYPROVIDER_API_KEY")
    if myprovider_key:
        m = MyProviderModel(myprovider_key, model=os.getenv("MYPROVIDER_MODEL", "my-default-model"))
        if m.is_available():
            self.models["myprovider"] = m
```

---

### Step 3 — Add to .env.example

```ini
# My Provider
# Get key: https://myprovider.com/api
MYPROVIDER_API_KEY=
MYPROVIDER_MODEL=my-default-model
```

---

### Step 4 — Add optional dependency to pyproject.toml

```toml
[project.optional-dependencies]
ai = [
    "google-genai>=1.0.0",
    "openai>=1.0.0",
    "anthropic>=0.25.0",
    "myprovider-sdk>=1.0.0",   # ← add this
]
```

---

## Adding a New Tool to an Existing Module

If you want to add a new sub-tool to an existing module (e.g. a new scanner):

1. Add the function to the relevant `ptcenter/modules/*.py` file following the same pattern.
2. Add the tool binary to `_PKG_MAP` in `sysinfo.py`.
3. Add the new menu option to `ui/menu.py` (increment the option count).
4. Add the `elif choice == "N": module.new_function(...)` case in `app.py`.
5. Update `docs/MODULES.md`.

---

## Core Principles

These must be respected in all contributions:

### No `shell=True`
Every external command must be constructed as a `list[str]` and passed to `run_command()` or `stream_command()`. No raw string concatenation for subprocess calls.

```python
# ✓ Correct
run_command(["nmap", "-sV", target])

# ✗ Wrong — shell injection risk
subprocess.run(f"nmap -sV {target}", shell=True)
```

### Validate all external input
Any value that came from user input, a file, or the network must be validated before use.

```python
# ✓ Correct
if not validate_target(target):
    console.print("[red]✗[/] Invalid target")
    return

# ✗ Wrong — unvalidated input passed to subprocess
run_command(["nmap", user_input])
```

### Distro-aware install hints
Never hardcode `apt install` or `pacman -S`. Use `install_hint(tool_name)` from `core.sysinfo` — it returns the right command for the running distro.

### Rich for all output
Use `console.print()` with Rich markup for all user-facing output. No bare `print()` calls.

### Consistent output file naming
All output files must follow the pattern: `<module>_<optional_target>_<timestamp>.txt` where `_ts()` provides the timestamp string `YYYYMMDD_HHMMSS`.

---

## Pull Request Guidelines

1. **One feature per PR** — keep PRs focused and reviewable.
2. **Test on at least one distro** before submitting (Arch/Kali preferred).
3. **Update docs** — add your feature to `docs/MODULES.md` and `README.md` if applicable.
4. **No secrets** — verify there are no API keys, tokens, or passwords in your commits. Run `git log --oneline -5` and `git diff HEAD` before pushing.
5. **Commit messages** — use the format `module: brief description`, e.g. `scanner: add masscan integration`.

```bash
# Before opening a PR
git diff main...feature/my-new-thing   # review your changes
grep -rn "API_KEY\|api_key\|password\|secret" ptcenter/  # check for secrets
```

---

## Bug Reports & Feature Requests

Open an issue on GitHub: [https://github.com/j0yb0y-m/ptcenter/issues](https://github.com/j0yb0y-m/ptcenter/issues)

For bug reports, include:
- Your distro and Python version (`python --version`)
- ptCenter version (`ptcenter --version`)
- The full error output / traceback
- Steps to reproduce

For feature requests, describe the tool or capability and why it fits ptCenter's scope.
