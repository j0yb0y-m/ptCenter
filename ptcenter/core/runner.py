"""
ptcenter.core.runner
====================
Safe subprocess wrapper.  All tool invocations in ptcenter must go through
run_command() or stream_command().  Neither ever uses shell=True with raw
user input.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

console = Console()
logger = logging.getLogger(__name__)


# Core runner

def run_command(
    command: list[str],
    output_file: Optional[str] = None,
    timeout: int = 300,
    env: Optional[dict] = None,
) -> tuple[bool, str]:
    """
    Execute *command* (a list of strings) safely — shell=False always.

    Args:
        command:     Command as a list[str], e.g. ["nmap", "-sV", "10.0.0.1"].
        output_file: Optional path to save stdout.
        timeout:     Seconds before the process is killed.
        env:         Optional environment dict to pass to the subprocess.

    Returns:
        (success: bool, output: str)
    """
    display = " ".join(command)
    console.print(f"[bold blue]▶[/] [dim]{display}[/dim]")

    try:
        result = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        combined = result.stdout
        if result.stderr:
            combined_with_err = result.stdout + "\n\n=== STDERR ===\n" + result.stderr
        else:
            combined_with_err = result.stdout

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as fh:
                fh.write(combined_with_err)

        if result.returncode == 0:
            console.print("[bold green]✓[/] Command completed")
            return True, combined
        else:
            console.print(f"[bold red]✗[/] Exit code {result.returncode}")
            logger.error("Command failed: %s\n%s", display, result.stderr)
            return False, result.stderr

    except subprocess.TimeoutExpired:
        msg = f"Command timed out after {timeout}s"
        console.print(f"[bold red]✗[/] {msg}")
        logger.error("Timeout: %s", display)
        return False, msg

    except FileNotFoundError:
        msg = f"Executable not found: {command[0]!r}"
        console.print(f"[bold red]✗[/] {msg}")
        logger.error(msg)
        return False, msg

    except Exception as exc:
        msg = f"Execution error: {exc}"
        console.print(f"[bold red]✗[/] {msg}")
        logger.error("Command execution error: %s", exc)
        return False, msg


def stream_command(
    command: list[str],
    output_file: Optional[str] = None,
    timeout: int = 600,
    env: Optional[dict] = None,
) -> tuple[bool, str]:
    """
    Stream subprocess output line-by-line in real time using Rich Live.
    Returns (success, full_output) after the process exits.
    """
    display = " ".join(command)
    console.print(f"[bold blue]▶[/] [dim]{display}[/dim]")
    lines: list[str] = []

    try:
        process = subprocess.Popen(
            command,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            out_fh = open(output_file, "w")
        else:
            out_fh = None

        spinner = Spinner("dots", text=Text(f"Running {command[0]}…", style="dim"))
        with Live(spinner, refresh_per_second=10, console=console):
            assert process.stdout is not None
            for line in process.stdout:
                line_stripped = line.rstrip("\n")
                lines.append(line_stripped)
                console.print(line_stripped)
                if out_fh:
                    out_fh.write(line)

        if out_fh:
            out_fh.close()

        process.wait(timeout=timeout)
        full_output = "\n".join(lines)

        if process.returncode == 0:
            console.print("[bold green]✓[/] Command completed")
            return True, full_output
        else:
            console.print(f"[bold red]✗[/] Exit code {process.returncode}")
            return False, full_output

    except subprocess.TimeoutExpired:
        msg = f"Command timed out after {timeout}s"
        console.print(f"[bold red]✗[/] {msg}")
        return False, msg

    except FileNotFoundError:
        msg = f"Executable not found: {command[0]!r}"
        console.print(f"[bold red]✗[/] {msg}")
        return False, msg

    except Exception as exc:
        msg = f"Streaming error: {exc}"
        console.print(f"[bold red]✗[/] {msg}")
        logger.error("Stream error: %s", exc)
        return False, msg
