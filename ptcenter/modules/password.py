"""
ptcenter.modules.password
=========================
Password attack module: hash identification, cracking (hashcat/john),
service brute force (hydra), password spray, wordlist manager, AI hash analysis.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from ptcenter.core.sysinfo import install_hint, rockyou_path, wordlist_dirs
from ptcenter.core.runner import run_command, stream_command
from ptcenter.core.validator import validate_ip, validate_port, validate_url

console = Console()


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _check(tool: str) -> bool:
    return shutil.which(tool) is not None


# Hash Identification
def hash_identification(output_dir: Path, ai_manager: Any = None) -> None:
    hash_val = console.input("[bold white][+] Paste hash: [/]").strip()
    if not hash_val:
        return

    identified = False

    if _check("hashid"):
        ok, res = run_command(["hashid", hash_val], timeout=10)
        if ok and res:
            console.print(res)
            identified = True
    elif _check("hash-identifier"):
        ok, res = run_command(["hash-identifier", hash_val], timeout=10)
        if ok and res:
            console.print(res)
            identified = True

    if not identified and ai_manager and ai_manager.is_available():
        console.print("[yellow]⚠[/] hashid/hash-identifier not installed — using AI fallback…")
        analysis = ai_manager.generate(
            f"Identify this hash type and suggest cracking strategies:\n{hash_val}",
            "You are an expert in password cracking and hash analysis. Be specific and technical, and provide confidence levels for your identification. If you can't identify, say so clearly.",
        )
        if analysis:
            ai_manager.display_analysis(analysis, "Hash Identification")
    elif not identified:
        console.print("[yellow]⚠[/] Install: [dim]uv add hashid[/]")


# Hash Cracking

def hash_cracking(output_dir: Path) -> None:
    console.print(
        "\n[bold blue]Tools:[/]\n"
        "  [magenta]1[/] hashcat\n"
        "  [magenta]2[/] john the ripper\n"
    )
    choice = console.input("[bold white][+] Tool: [/]").strip()

    hash_val = console.input("[bold white][+] Hash (or file path): [/]").strip()
    wordlist = console.input("[bold white][+] Wordlist [/usr/share/wordlists/rockyou.txt]: [/]").strip()
    wordlist = wordlist or rockyou_path()

    if not Path(wordlist).exists():
        console.print(f"[yellow]⚠[/] Wordlist not found: {wordlist}")
        console.print(f"[dim]Tip: On BlackArch/Kali: sudo pacman -S wordlists  |  sudo gunzip {rockyou_path()}.gz[/]")

    ts = _ts()
    out = str(output_dir / f"cracked_{ts}.txt")

    if choice == "1":
        if not _check("hashcat"):
            console.print(f"[red]✗[/] hashcat not installed.  [dim]{install_hint('hashcat')}[/]")
            return
        console.print(
            "\n[bold blue]Attack modes:[/]\n"
            "  [magenta]0[/] Straight (dictionary)\n"
            "  [magenta]3[/] Brute-force mask\n"
        )
        mode = console.input("[bold white][+] Attack mode [0]: [/]").strip() or "0"
        hash_type = console.input("[bold white][+] Hash type (-m, e.g. 0=MD5, 100=SHA1, 1800=sha512crypt): [/]").strip() or "0"

        rules: list[str] = []
        if console.input("[bold white][?] Apply rules? (y/n): [/]").strip().lower() == "y":
            rule = console.input("[bold white][+] Rule file (e.g. best64.rule, Enter to skip): [/]").strip()
            if rule:
                rules = ["-r", rule]

        is_file = Path(hash_val).exists()
        hash_arg = hash_val if is_file else hash_val

        cmd = (
            ["hashcat", "-m", hash_type, "-a", mode, hash_arg, wordlist]
            + rules
            + ["--outfile", out, "--force"]
        )
        stream_command(cmd, timeout=3600)

        if Path(out).exists():
            console.print(f"\n[bold green]✓[/] Cracked passwords in: [cyan]{out}[/]")
            with open(out) as fh:
                console.print(fh.read())

    elif choice == "2":
        if not _check("john"):
            console.print(f"[red]✗[/] john not installed.  [dim]{install_hint('john')}[/]")
            return

        is_file = Path(hash_val).exists()
        if not is_file:
            # Write hash to temp file
            tmp = output_dir / f"john_hash_{ts}.txt"
            tmp.write_text(hash_val + "\n")
            hash_arg = str(tmp)
        else:
            hash_arg = hash_val

        cmd = ["john", f"--wordlist={wordlist}", hash_arg]
        stream_command(cmd, timeout=3600)
        # Show results
        run_command(["john", "--show", hash_arg], out, timeout=10)
        console.print(f"[bold green]✓[/] Results: [cyan]{out}[/]")
    else:
        console.print("[red]✗[/] Invalid selection")


# Service Brute Force (hydra)

_HYDRA_PROTOCOLS = {
    "1": "ssh", "2": "ftp", "3": "http-post-form", "4": "smb",
    "5": "rdp", "6": "mysql", "7": "postgres", "8": "telnet", "9": "smtp",
}


def service_brute_force(output_dir: Path) -> None:
    if not _check("hydra"):
        console.print(f"[red]✗[/] hydra not installed.  [dim]{install_hint('hydra')}[/]")
        return

    target = console.input("[bold white][+] Target IP/host: [/]").strip()
    if not target:
        return

    console.print("\n[bold blue]Protocols:[/]")
    for k, v in _HYDRA_PROTOCOLS.items():
        console.print(f"  [magenta]{k}[/] {v}")
    proto_choice = console.input("[bold white][+] Protocol: [/]").strip()

    if proto_choice not in _HYDRA_PROTOCOLS:
        console.print("[red]✗[/] Invalid selection")
        return

    protocol = _HYDRA_PROTOCOLS[proto_choice]
    port = console.input(f"[bold white][+] Port (Enter for default): [/]").strip()
    port_flag: list[str] = ["-s", port] if port and validate_port(port) else []

    username_input = console.input("[bold white][+] Username (or -L for file path): [/]").strip()
    password_input = console.input("[bold white][+] Password file [rockyou.txt]: [/]").strip()
    password_input = password_input or rockyou_path()

    if not Path(password_input).exists():
        console.print(f"[yellow]⚠[/] Password file not found: {password_input}")

    ts = _ts()
    out = str(output_dir / f"hydra_{protocol}_{ts}.txt")

    user_flag: list[str]
    if username_input.startswith("/") or Path(username_input).exists():
        user_flag = ["-L", username_input]
    else:
        user_flag = ["-l", username_input]

    base_cmd = ["hydra"] + port_flag + user_flag + ["-P", password_input, target, protocol, "-o", out]

    # HTTP-POST-FORM needs extra args
    if protocol == "http-post-form":
        form_url = console.input("[bold white][+] Form URL path (e.g. /login): [/]").strip()
        post_params = console.input("[bold white][+] POST params (e.g. user=^USER^&pass=^PASS^): [/]").strip()
        fail_string = console.input("[bold white][+] Failure string: [/]").strip()
        base_cmd = ["hydra"] + port_flag + user_flag + ["-P", password_input, target,
                    f"http-post-form", f"{form_url}:{post_params}:{fail_string}", "-o", out]

    stream_command(base_cmd, timeout=3600)
    console.print(f"[bold green]✓[/] Results: [cyan]{out}[/]")


# Password Spray

def password_spray(output_dir: Path) -> None:
    target = console.input("[bold white][+] Target IP/host: [/]").strip()
    if not target:
        return

    password = console.input("[bold white][+] Single password to spray: [/]").strip()
    user_list = console.input("[bold white][+] Username list file: [/]").strip()
    if not Path(user_list).exists():
        console.print(f"[red]✗[/] File not found: {user_list}")
        return

    delay = console.input("[bold white][+] Delay between attempts (secs) [30]: [/]").strip() or "30"
    proto_choice = console.input(
        "[bold white][+] Protocol — 1=SSH, 2=SMB, 3=RDP [1]: [/]"
    ).strip() or "1"
    proto_map = {"1": "ssh", "2": "smb", "3": "rdp"}
    protocol = proto_map.get(proto_choice, "ssh")

    if protocol in ("smb", "rdp") and _check("crackmapexec"):
        console.print("[bold blue]▶[/] Using crackmapexec for spray…")
        cmd = ["crackmapexec", protocol, target, "-u", user_list, "-p", password,
               "--continue-on-success"]
    elif _check("hydra"):
        console.print("[bold blue]▶[/] Using hydra for spray…")
        cmd = ["hydra", "-L", user_list, "-p", password, target, protocol]
    else:
        console.print("[red]✗[/] Neither crackmapexec nor hydra installed")
        return

    ts = _ts()
    out = str(output_dir / f"spray_{protocol}_{ts}.txt")
    console.print(f"[yellow]⚠[/] Spraying with delay={delay}s — check account lockout policies!")
    stream_command(cmd, out, timeout=7200)
    console.print(f"[bold green]✓[/] Results: [cyan]{out}[/]")


# Wordlist Manager

def wordlist_manager(output_dir: Path) -> None:
    console.print(
        "\n[bold blue]Options:[/]\n"
        "  [magenta]1[/] List available wordlists\n"
        "  [magenta]2[/] Download SecLists\n"
        "  [magenta]3[/] Generate custom wordlist (crunch)\n"
    )
    choice = console.input("[bold white][+] Select: [/]").strip()

    if choice == "1":
        search_dirs = wordlist_dirs()
        table = Table(title="[bold green]Available Wordlists[/]")
        table.add_column("Name")
        table.add_column("Size", justify="right")
        table.add_column("Lines", justify="right")

        for d in search_dirs:
            if not d.exists():
                continue
            for f in sorted(d.rglob("*.txt"))[:50]:
                try:
                    size = f.stat().st_size
                    lines = sum(1 for _ in open(f, errors="ignore"))
                    table.add_row(
                        str(f.relative_to(d.parent)),
                        f"{size // 1024} KB",
                        str(lines),
                    )
                except Exception:
                    table.add_row(str(f), "?", "?")

        console.print(table)

    elif choice == "2":
        dest = Path.home() / "wordlists" / "SecLists"
        if dest.exists():
            console.print(f"[yellow]⚠[/] SecLists already at: {dest}")
        else:
            console.print("[bold blue]▶[/] Cloning SecLists (this may take a while)…")
            run_command(
                ["git", "clone", "--depth", "1",
                 "https://github.com/danielmiessler/SecLists.git", str(dest)],
                timeout=600,
            )
            console.print(f"[bold green]✓[/] SecLists downloaded to: {dest}")

    elif choice == "3":
        if not _check("crunch"):
            console.print(f"[red]✗[/] crunch not installed.  [dim]{install_hint('crunch')}[/]")
            return
        min_len = console.input("[bold white][+] Min length: [/]").strip()
        max_len = console.input("[bold white][+] Max length: [/]").strip()
        charset = console.input("[bold white][+] Charset (e.g. abc123): [/]").strip()
        ts = _ts()
        out = str(output_dir / f"crunch_{ts}.txt")
        run_command(["crunch", min_len, max_len, charset, "-o", out], timeout=300)
        console.print(f"[bold green]✓[/] Generated: [cyan]{out}[/]")
    else:
        console.print("[red]✗[/] Invalid selection")


# AI Hash Analysis

def ai_hash_analysis(output_dir: Path, ai_manager: Any) -> None:
    if not ai_manager or not ai_manager.is_available():
        console.print("[red]✗[/] No AI model available")
        return

    hash_val = console.input("[bold white][+] Paste hash for analysis: [/]").strip()
    if not hash_val:
        return

    prompt = (
        f"Analyse this hash: {hash_val}\n\n"
        "1. Identify the hash type (with confidence)\n"
        "2. Suggest the best cracking strategy (dictionary / brute / rainbow / hybrid)\n"
        "3. Estimate crack time on: GTX 1080, RTX 3090, dedicated cracking rig\n"
        "4. List relevant rainbow table resources\n"
        "5. Provide the exact hashcat command to crack it with rockyou.txt\n"
        "6. Any additional hints from the hash structure"
    )

    analysis = ai_manager.generate(
        prompt,
        "You are an expert in password cracking and hash analysis. Be specific and technical.",
    )
    if analysis:
        ai_manager.display_analysis(analysis, "Hash Analysis")
        ts = _ts()
        out = output_dir / f"hash_analysis_{ts}.txt"
        out.write_text(f"Hash: {hash_val}\n\n{analysis}")
        console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")
