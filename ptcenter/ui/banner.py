"""ptcenter.ui.banner — display_banner()."""

from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ptcenter.core.sysinfo import detect_family, distro_name

console = Console()


def display_banner(
    ai_model_name: str,
    ai_available: bool,
    model_count: int,
    output_dir: Path,
    version: str = "2.0",
) -> None:
    width = shutil.get_terminal_size().columns

    art = Text(justify="center")
    art.append(
        r"""
 ██████╗ ████████╗    ██████╗███████╗███╗   ██╗████████╗███████╗██████╗
 ██╔══██╗╚══██╔══╝   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗
 ██████╔╝   ██║      ██║     █████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝
 ██╔═══╝    ██║      ██║     ██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗
 ██║        ██║      ╚██████╗███████╗██║ ╚████║   ██║   ███████╗██║  ██║
 ╚═╝        ╚═╝       ╚═════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
""",
        style="bold cyan",
    )

    ai_line = (
        f"[bold green]✓ {ai_model_name}[/]"
        if ai_available
        else "[bold red]✗ Disabled — add an API key to .env[/]"
    )
    model_count_str = (
        f"[dim]({model_count} model{'s' if model_count != 1 else ''} loaded)[/]"
        if model_count
        else ""
    )

    family = detect_family()
    distro = distro_name()
    pkg_mgr = {"arch": "pacman/AUR", "debian": "apt", "fedora": "dnf"}.get(family, "unknown")

    meta = (
        f"[dim]Developer:[/] Mahdi (@j0yb0y-m)    "
        f"[dim]Version:[/] [cyan]{version}[/]    "
        f"[dim]AI:[/] {ai_line} {model_count_str}\n"
        f"[dim]Distro:[/]  [cyan]{distro}[/]  [dim]·[/]  [cyan]{pkg_mgr}[/]    "
        f"[dim]Output:[/] [cyan]{output_dir}[/]"
    )

    console.print(art)
    console.print(
        Panel(
            meta,
            title="[bold cyan]🔒Penetration Testing Centre 🔒[/]",
            border_style="cyan",
            expand=False,
            padding=(0, 2),
        )
    )
