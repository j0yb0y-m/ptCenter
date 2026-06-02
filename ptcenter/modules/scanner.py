"""
ptcenter.modules.scanner
========================
All scanning functions: nmap, subdomain, dir brute, nikto, ssl, dns, smb,
agentic recon.  All subprocess calls go through core.runner — shell=False.
"""

from __future__ import annotations

import shlex
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ptcenter.core.runner import run_command, stream_command
from ptcenter.core.sysinfo import install_hint
from ptcenter.core.validator import (
    is_in_scope,
    sanitize_nmap_flags,
    validate_domain,
    validate_target,
    validate_url,
)
from ptcenter.ui import menu as ui

console = Console()


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _check(tool: str) -> bool:
    return shutil.which(tool) is not None


def _scope_check(target: str, session: Optional[dict], output_dir: Path) -> bool:
    """Return True if the target is in scope (or no session active)."""
    if not session:
        return True
    scope = session.get("scope", [])
    if not scope:
        return True
    if not is_in_scope(target, scope):
        console.print(
            f"[bold yellow]⚠[/] [yellow]{target}[/] is [bold red]OUT OF SCOPE[/] "
            f"for this session.  Scope: {scope}"
        )
        confirm = console.input(
            "[bold yellow]Type 'CONFIRM' to run anyway, or press Enter to cancel: [/]"
        ).strip()
        return confirm == "CONFIRM"
    return True



# Nmap

NMAP_PROFILES: dict[str, str] = {
    "1": "-F -T4",
    "2": "-p- -T4",
    "3": "-sV -T4",
    "4": "-O -T4",
    "5": "-A -T4",
    "6": "-sS -T4",
    "7": "-sU -T4",
    "8": "--script vuln -T4",
    "9": "-sC -sV -T4",  # NSE custom (will ask for categories)
    "10": "-sC -sV -T4",  # custom
}


def nmap_scan(
    output_dir: Path,
    session: Optional[dict] = None,
    ai_manager: Any = None,
    auto_ai: bool = True,
) -> Optional[str]:
    if not _check("nmap"):
        console.print(f"[bold red]✗[/] nmap not installed.  [dim]{install_hint('nmap')}[/]")
        return None

    target = console.input("[bold white][+] Target IP / Domain / Range: [/]").strip()
    if not target:
        console.print("[red]✗[/] Target cannot be empty")
        return None
    if not validate_target(target) and "/" not in target:
        console.print(f"[bold red]✗[/] Invalid target: {target!r}")
        return None
    if not _scope_check(target, session, output_dir):
        return None

    console.print(
        "\n[bold blue]Scan Profiles:[/]\n"
        "  [magenta]1[/] Quick Scan (Top 100 ports)\n"
        "  [magenta]2[/] Full TCP Scan (-p-)\n"
        "  [magenta]3[/] Service Detection (-sV)\n"
        "  [magenta]4[/] OS Detection (-O, needs root)\n"
        "  [magenta]5[/] Aggressive Scan (-A)\n"
        "  [magenta]6[/] Stealth SYN (-sS, needs root)\n"
        "  [magenta]7[/] UDP Scan (-sU, needs root)\n"
        "  [magenta]8[/] Vuln Script (--script vuln)\n"
        "  [magenta]9[/] NSE Script Categories (choose categories)\n"
        " [magenta]10[/] Custom flags\n"
    )
    profile = console.input("[bold white][+] Select scan profile [1-10]: [/]").strip()

    if profile == "9":
        cats = console.input(
            "[bold white][+] Script categories (comma-sep, e.g. auth,safe,vuln): [/]"
        ).strip() or "default"
        nmap_flags_str = f"--script={cats} -T4"
        try:
            flag_tokens = sanitize_nmap_flags(nmap_flags_str)
        except ValueError as exc:
            console.print(f"[red]✗[/] {exc}")
            return None
    elif profile == "10":
        raw = console.input("[bold white][+] Custom nmap flags: [/]").strip()
        try:
            flag_tokens = sanitize_nmap_flags(raw)
        except ValueError as exc:
            console.print(f"[red]✗[/] {exc}")
            return None
    else:
        flags_str = NMAP_PROFILES.get(profile, "-sC -sV -T4")
        flag_tokens = shlex.split(flags_str)

    ts = _ts()
    safe_target = target.replace("/", "_")
    out_txt = str(output_dir / f"nmap_{safe_target}_{ts}.txt")
    out_xml = str(output_dir / f"nmap_{safe_target}_{ts}.xml")

    cmd = ["nmap"] + flag_tokens + ["-oN", out_txt, "-oX", out_xml, target]

    console.print(Panel(f"[dim]Starting nmap scan of [cyan]{target}[/][/dim]", border_style="cyan"))
    success, result = stream_command(cmd, timeout=600)

    if not success:
        return None

    # Parse XML for structured findings
    open_ports: list[dict] = []
    try:
        tree = ET.parse(out_xml)
        root = tree.getroot()
        for host in root.findall("host"):
            for port_el in host.findall(".//port"):
                state = port_el.find("state")
                if state is not None and state.get("state") == "open":
                    svc = port_el.find("service")
                    open_ports.append({
                        "port": port_el.get("portid", "?"),
                        "protocol": port_el.get("protocol", "tcp"),
                        "service": svc.get("name", "") if svc is not None else "",
                        "version": (
                            f"{svc.get('product', '')} {svc.get('version', '')}".strip()
                            if svc is not None else ""
                        ),
                    })
    except Exception:
        pass

    if open_ports:
        table = Table(title="[bold green]Open Ports[/]", show_lines=False)
        table.add_column("Port", style="cyan", width=8)
        table.add_column("Proto", width=6)
        table.add_column("Service", width=14)
        table.add_column("Version")
        for p in open_ports:
            table.add_row(p["port"], p["protocol"], p["service"], p["version"])
        console.print(table)

    # AI analysis
    analysis = None
    if auto_ai and ai_manager and ai_manager.is_available():
        try:
            with open(out_txt) as fh:
                scan_text = fh.read()
        except Exception:
            scan_text = result
        analysis = ai_manager.analyze_scan(scan_text, "Nmap")
        if analysis:
            ai_manager.display_analysis(analysis)

    # Add session finding
    if session is not None and open_ports:
        from ptcenter.core.session import add_finding
        title = f"Nmap found {len(open_ports)} open port(s) on {target}"
        details = (analysis or result)[:3000]
        add_finding(session, "scanner", "nmap", "Info", title, details, out_txt)

    console.print(f"[bold green]✓[/] Saved: [cyan]{out_txt}[/]")
    return out_txt



# Subdomain discovery

def subdomain_scan(
    output_dir: Path,
    session: Optional[dict] = None,
    ai_manager: Any = None,
) -> None:
    has_sublist3r = _check("sublist3r")
    has_amass = _check("amass")
    has_subfinder = _check("subfinder")

    if not (has_sublist3r or has_amass or has_subfinder):
        console.print("[red]✗[/] No subdomain tools installed.")
        console.print(f"[dim]{install_hint('sublist3r')}  |  {install_hint('amass')}  |  {install_hint('subfinder')}[/]")
        return

    domain = console.input("[bold white][+] Domain: [/]").strip()
    if not domain or not validate_domain(domain):
        console.print("[red]✗[/] Invalid domain")
        return

    ts = _ts()
    out = output_dir / f"subdomains_{domain}_{ts}.txt"

    tools_avail = []
    if has_sublist3r:
        tools_avail.append(("1", "Sublist3r"))
    if has_amass:
        tools_avail.append(("2", "Amass"))
    if has_subfinder:
        tools_avail.append(("3", "Subfinder"))
    tools_avail.append(("4", "All available"))

    for num, name in tools_avail:
        console.print(f"  [magenta]{num}[/] - {name}")
    choice = console.input("[bold white][+] Select: [/]").strip()

    all_subs: set[str] = set()

    def _run_tool(cmd: list[str], out_file: str) -> None:
        ok, res = run_command(cmd, out_file, timeout=300)
        if ok and Path(out_file).exists():
            with open(out_file) as fh:
                all_subs.update(line.strip() for line in fh if line.strip())

    if choice in ("1", "4") and has_sublist3r:
        _run_tool(["sublist3r", "-d", domain, "-o", str(out) + ".sl3"], str(out) + ".sl3")
    if choice in ("2", "4") and has_amass:
        _run_tool(["amass", "enum", "-passive", "-d", domain, "-o", str(out) + ".amass"], str(out) + ".amass")
    if choice in ("3", "4") and has_subfinder:
        _run_tool(["subfinder", "-d", domain, "-o", str(out) + ".sf"], str(out) + ".sf")

    if all_subs:
        with open(out, "w") as fh:
            fh.write("\n".join(sorted(all_subs)) + "\n")

        table = Table(title=f"[bold green]Subdomains for {domain}[/]", show_lines=False)
        table.add_column("Subdomain")
        for sub in sorted(all_subs)[:50]:
            table.add_row(sub)
        if len(all_subs) > 50:
            table.add_row(f"[dim]… and {len(all_subs) - 50} more[/]")
        console.print(table)
        console.print(f"[bold green]✓[/] {len(all_subs)} subdomains saved: [cyan]{out}[/]")

        if session is not None:
            from ptcenter.core.session import add_finding
            add_finding(session, "scanner", "subdomain", "Info",
                        f"Found {len(all_subs)} subdomains for {domain}",
                        "\n".join(sorted(all_subs)[:200]), str(out))
    else:
        console.print("[yellow]⚠[/] No subdomains found")



# Directory brute force


def directory_brute_force(
    output_dir: Path,
    session: Optional[dict] = None,
) -> None:
    has_gobuster = _check("gobuster")
    has_dirb = _check("dirb")
    has_dirsearch = _check("dirsearch")

    if not (has_gobuster or has_dirb or has_dirsearch):
        console.print(f"[red]✗[/] No dir brute tools installed.  [dim]{install_hint('gobuster')}  |  {install_hint('dirb')}[/]")
        return

    url = console.input("[bold white][+] Target URL: [/]").strip()
    if not url or not validate_url(url):
        console.print("[red]✗[/] Invalid URL — must start with http:// or https://")
        return

    ts = _ts()
    out = str(output_dir / f"directories_{ts}.txt")
    wordlist = "/usr/share/wordlists/dirb/common.txt"

    options: list[str] = []
    if has_gobuster:
        options.append("1 - Gobuster (fast)")
    if has_dirb:
        options.append("2 - Dirb (classic)")
    if has_dirsearch:
        options.append("3 - Dirsearch")
    for o in options:
        console.print(f"  [magenta]{o}[/]")
    choice = console.input("[bold white][+] Tool: [/]").strip()

    if choice == "1" and has_gobuster:
        custom_wl = console.input(f"[bold white][+] Wordlist [{wordlist}]: [/]").strip() or wordlist
        cmd = ["gobuster", "dir", "-u", url, "-w", custom_wl, "-o", out, "-q"]
    elif choice == "2" and has_dirb:
        cmd = ["dirb", url, wordlist, "-o", out]
    elif choice == "3" and has_dirsearch:
        cmd = ["dirsearch", "-u", url, "-o", out]
    else:
        console.print("[red]✗[/] Invalid / unavailable selection")
        return

    stream_command(cmd, timeout=600)
    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")


# Nikto

def nikto_scan(
    output_dir: Path,
    session: Optional[dict] = None,
    ai_manager: Any = None,
    auto_ai: bool = True,
) -> None:
    if not _check("nikto"):
        console.print(f"[red]✗[/] nikto not installed.  [dim]{install_hint('nikto')}[/]")
        return

    target = console.input("[bold white][+] Target URL / IP: [/]").strip()
    if not target:
        console.print("[red]✗[/] Target cannot be empty")
        return
    if not _scope_check(target, session, output_dir):
        return

    ts = _ts()
    out = str(output_dir / f"nikto_{ts}.txt")

    cmd = ["nikto", "-host", target, "-o", out]
    success, result = stream_command(cmd, timeout=900)

    if success and ai_manager and ai_manager.is_available() and auto_ai:
        try:
            with open(out) as fh:
                content = fh.read()
        except Exception:
            content = result
        analysis = ai_manager.analyze_scan(content, "Nikto Web")
        if analysis:
            ai_manager.display_analysis(analysis)
            if session is not None:
                from ptcenter.core.session import add_finding
                add_finding(session, "scanner", "nikto", "Medium",
                            f"Nikto web scan on {target}", analysis[:3000], out)

    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")



# SSL / TLS

def ssl_scan(output_dir: Path, ai_manager: Any = None) -> None:
    target = console.input("[bold white][+] Domain / IP: [/]").strip()
    if not target:
        return

    ts = _ts()
    out = str(output_dir / f"ssl_{ts}.txt")

    if _check("testssl.sh"):
        cmd = ["testssl.sh", target]
        stream_command(cmd, out, timeout=300)
    elif _check("sslscan"):
        cmd = ["sslscan", target]
        run_command(cmd, out, timeout=120)
    else:
        # Pure-Python fallback
        console.print("[yellow]⚠[/] No sslscan/testssl.sh — running Python SSL check…")
        _python_ssl_check(target, out)

    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")


def _python_ssl_check(host: str, out_file: str) -> None:
    import ssl
    import socket
    import datetime as dt

    result_lines: list[str] = []
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.create_connection((host, 443), timeout=10), server_hostname=host) as s:
            cert = s.getpeercert()
            cipher = s.cipher()
            version = s.version()

        expiry_str = cert.get("notAfter", "")
        expiry = dt.datetime.strptime(expiry_str, "%b %d %H:%M:%S %Y %Z") if expiry_str else None
        cn = dict(x[0] for x in cert.get("subject", []) if x).get("commonName", "?")
        sans = [v for _, v in cert.get("subjectAltName", [])]

        result_lines += [
            f"Host       : {host}:443",
            f"Protocol   : {version}",
            f"Cipher     : {cipher[0] if cipher else '?'}",
            f"CN         : {cn}",
            f"SANs       : {', '.join(sans[:10])}",
            f"Expires    : {expiry_str}",
        ]
        if expiry:
            days_left = (expiry - dt.datetime.utcnow()).days
            result_lines.append(
                f"Days left  : {days_left} {'[EXPIRED]' if days_left < 0 else ''}"
            )

        table = Table(title="[bold green]SSL/TLS Summary[/]")
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        for line in result_lines:
            if ":" in line:
                k, v = line.split(":", 1)
                table.add_row(k.strip(), v.strip())
        console.print(table)

    except Exception as exc:
        result_lines.append(f"Error: {exc}")
        console.print(f"[red]✗[/] SSL check failed: {exc}")

    with open(out_file, "w") as fh:
        fh.write("\n".join(result_lines))


# DNS enumeration

def dns_enumeration(output_dir: Path, session: Optional[dict] = None) -> None:
    domain = console.input("[bold white][+] Domain: [/]").strip()
    if not domain or not validate_domain(domain):
        console.print("[red]✗[/] Invalid domain")
        return

    ts = _ts()
    out = output_dir / f"dns_{domain}_{ts}.txt"
    lines: list[str] = []

    for rtype in ["A", "AAAA", "MX", "NS", "TXT", "SOA", "CNAME"]:
        ok, res = run_command(["dig", domain, rtype, "+short"], timeout=15)
        if ok and res.strip():
            lines.append(f"\n=== {rtype} Records ===\n{res}")
            console.print(f"[cyan]{rtype}[/]: {res.strip()}")

    # Zone transfer
    ok, ns_out = run_command(["dig", domain, "NS", "+short"], timeout=10)
    ns_servers = [s.rstrip(".") for s in ns_out.splitlines() if s.strip()] if ok else []
    for ns in ns_servers:
        ok, axfr = run_command(["dig", "axfr", f"@{ns}", domain], timeout=20)
        if ok and "XFR size" in axfr:
            lines.append(f"\n=== AXFR from {ns} ===\n{axfr}")
            console.print(f"[bold green]✓[/] Zone transfer succeeded from {ns}")

    with open(out, "w") as fh:
        fh.write(f"DNS Enumeration: {domain}\n{'='*60}\n")
        fh.write("\n".join(lines))

    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")


# SMB enumeration

def smb_enumeration(output_dir: Path, session: Optional[dict] = None) -> None:
    target = console.input("[bold white][+] Target IP: [/]").strip()
    if not target:
        return

    ts = _ts()
    out = str(output_dir / f"smb_{target}_{ts}.txt")

    if _check("netexec"):
        cmd = ["netexec", "smb", target, "--shares", "-u", "", "-p", ""]
        run_command(cmd, out, timeout=120)
    elif _check("enum4linux"):
        cmd = ["enum4linux", "-a", target]
        run_command(cmd, out, timeout=600)
    else:
        console.print("[red]✗[/] Neither netexec nor enum4linux installed.")
        console.print(f"[dim]{install_hint('enum4linux')}  |  {install_hint('netexec')}[/]")
        return

    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")



# Agentic Recon

def agentic_recon(
    output_dir: Path,
    ai_manager: Any,
    session: Optional[dict] = None,
) -> None:
    if not ai_manager or not ai_manager.is_available():
        console.print("[red]✗[/] AI model not available for agentic recon")
        return

    target = console.input("[bold white][+] Target: [/]").strip()
    if not target or not validate_target(target):
        console.print("[red]✗[/] Invalid target")
        return
    if not _scope_check(target, session, output_dir):
        return

    max_iter = 5
    raw = console.input(f"[bold white][+] Max iterations [{max_iter}]: [/]").strip()
    if raw.isdigit():
        max_iter = int(raw)

    allowed_tools = {"nmap", "nikto", "gobuster", "sqlmap", "sslscan", "enum4linux", "netexec", "dirsearch", "gobuster", "metasploit", "feroxbuster", "tshark", "whatweb", "sublist3r", "amass", "subfinder", "routersploit", "wpscan", "hydra", "medusa", "nuclei", "masscan"}
    context = f"Target: {target}"
    last_output = ""

    # Initial nmap
    console.print(Panel(f"[cyan]Agentic Recon — starting with initial nmap of {target}[/]", border_style="cyan"))
    ts = _ts()
    init_out = str(output_dir / f"agentic_nmap_{ts}.txt")
    ok, last_output = stream_command(
        ["nmap", "-sC", "-sV", "-T4", target, "-oN", init_out], timeout=300
    )

    for iteration in range(1, max_iter + 1):
        console.rule(f"[bold cyan]Iteration {iteration}/{max_iter}[/]")
        decision = ai_manager.decide_next_action(last_output, context)

        if not decision:
            console.print("[yellow]⚠[/] AI could not decide next action — stopping")
            break

        tool = decision.get("tool", "")
        flags_raw = decision.get("flags", "")
        tgt = decision.get("target", target)
        reason = decision.get("reason", "")

        if tool not in allowed_tools:
            console.print(f"[yellow]⚠[/] AI requested unknown tool {tool!r} — skipping")
            break

        console.print(
            Panel(
                f"[bold]Tool:[/] [cyan]{tool}[/]  [bold]Target:[/] {tgt}\n"
                f"[bold]Flags:[/] [dim]{flags_raw}[/]\n"
                f"[bold]Reason:[/] {reason}",
                title="[bold cyan]AI Decision[/]",
                border_style="cyan",
            )
        )
        approve = console.input("[bold yellow]Approve? (y/n): [/]").strip().lower()
        if approve != "y":
            console.print("[dim]Skipped[/]")
            continue

        try:
            flag_tokens = sanitize_nmap_flags(flags_raw)
        except ValueError as exc:
            console.print(f"[red]✗[/] Unsafe flags: {exc}")
            continue

        ts2 = _ts()
        step_out = str(output_dir / f"agentic_{tool}_{ts2}.txt")
        _ok, last_output = stream_command([tool] + flag_tokens + [tgt], step_out, timeout=300)

        context += f"\nIteration {iteration}: ran {tool} against {tgt}."

        if session is not None:
            from ptcenter.core.session import add_finding
            add_finding(
                session, "scanner", tool, "Info",
                f"Agentic step {iteration}: {tool} on {tgt}",
                f"Reason: {reason}\n\nOutput:\n{last_output[:1500]}",
                step_out,
            )

    console.print("[bold green]✓[/] Agentic recon complete")
