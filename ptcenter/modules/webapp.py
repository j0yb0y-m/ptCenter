"""
ptcenter.modules.webapp
=======================
Web Application Testing: SQLMap, ffuf, dalfox, CORS check (pure-Python),
JWT analyser, headers security audit, tech stack fingerprinting.
"""

from __future__ import annotations

import base64
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ptcenter.core.sysinfo import install_hint
from ptcenter.core.runner import run_command, stream_command
from ptcenter.core.validator import validate_url

console = Console()


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _check(tool: str) -> bool:
    return shutil.which(tool) is not None


def _ask_url(prompt: str = "Target URL") -> Optional[str]:
    url = console.input(f"[bold white][+] {prompt}: [/]").strip()
    if not url or not validate_url(url):
        console.print("[red]✗[/] Invalid URL — must start with http:// or https://")
        return None
    return url


# SQLMap
def sqlmap_integration(output_dir: Path, ai_manager: Any = None) -> None:
    if not _check("sqlmap"):
        console.print(f"[red]✗[/] sqlmap not installed.  [dim]{install_hint('sqlmap')}[/]")
        return

    url = _ask_url()
    if not url:
        return

    cookie = console.input("[bold white][+] Cookie/session header (Enter to skip): [/]").strip()
    depth = console.input("[bold white][+] Crawl depth [0=off]: [/]").strip() or "0"
    level = console.input("[bold white][+] Level [2]: [/]").strip() or "2"
    risk = console.input("[bold white][+] Risk [1]: [/]").strip() or "1"

    ts = _ts()
    out = str(output_dir / f"sqlmap_{ts}.txt")

    cmd = ["sqlmap", "-u", url, "--batch", "--level", level, "--risk", risk]
    if cookie:
        cmd += ["--cookie", cookie]
    if depth != "0":
        cmd += ["--crawl", depth]
    cmd += ["-o", "--output-dir", str(output_dir / f"sqlmap_{ts}")]

    success, result = stream_command(cmd, out, timeout=900)

    if success and ai_manager and ai_manager.is_available():
        analysis = ai_manager.analyze_scan(result, "SQLMap")
        if analysis:
            ai_manager.display_analysis(analysis, "SQLMap Analysis")

    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")


# Parameter Fuzzing (ffuf)
def parameter_fuzzing(output_dir: Path) -> None:
    if not _check("ffuf"):
        console.print(f"[red]✗[/] ffuf not installed.  [dim]{install_hint('ffuf')}[/]")
        return

    console.print(
        "\n[bold blue]Modes:[/]\n"
        "  [magenta]1[/] Directory / file fuzzing (GET)\n"
        "  [magenta]2[/] Parameter fuzzing (POST body / query string)\n"
    )
    mode = console.input("[bold white][+] Mode: [/]").strip()

    url = console.input("[bold white][+] URL (use FUZZ as placeholder): [/]").strip()
    if not url:
        return

    wordlist = console.input("[bold white][+] Wordlist: [/]").strip()
    if not wordlist or not Path(wordlist).exists():
        wordlist = "/usr/share/wordlists/dirb/common.txt"
        console.print(f"[yellow]⚠[/] Using default wordlist: {wordlist}")

    codes = console.input("[bold white][+] Match status codes [200,301,302,403]: [/]").strip() or "200,301,302,403"
    filter_codes = console.input("[bold white][+] Filter status codes (Enter to skip): [/]").strip()

    ts = _ts()
    out = str(output_dir / f"ffuf_{ts}.json")

    cmd = ["ffuf", "-u", url, "-w", wordlist, "-mc", codes, "-o", out, "-of", "json", "-s"]
    if filter_codes:
        cmd += ["-fc", filter_codes]
    if mode == "2":
        method = console.input("[bold white][+] Method [POST]: [/]").strip() or "POST"
        cmd += ["-X", method]

    success, result = stream_command(cmd, timeout=600)

    # Parse JSON output
    if Path(out).exists():
        try:
            data = json.loads(Path(out).read_text())
            results = data.get("results", [])
            table = Table(title=f"[bold green]ffuf Results ({len(results)} hits)[/]")
            table.add_column("URL")
            table.add_column("Status", width=7)
            table.add_column("Size", width=8)
            table.add_column("Words", width=8)
            for r in sorted(results, key=lambda x: x.get("status", 0)):
                table.add_row(
                    r.get("url", ""),
                    str(r.get("status", "")),
                    str(r.get("length", "")),
                    str(r.get("words", "")),
                )
            console.print(table)
        except Exception:
            pass

    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")

# XSS Scanner (dalfox)

def xss_scanner(output_dir: Path) -> None:
    if not _check("dalfox"):
        console.print(f"[red]✗[/] dalfox not installed.  [dim]{install_hint('dalfox')}[/]")
        return

    url = _ask_url()
    if not url:
        return

    ts = _ts()
    out = str(output_dir / f"dalfox_{ts}.txt")

    cmd = ["dalfox", "url", url, "-o", out]
    stream_command(cmd, timeout=300)
    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")


# CORS Check (pure-Python)
def cors_check(output_dir: Path) -> None:
    url = _ask_url()
    if not url:
        return

    tests = [
        ("Arbitrary origin reflection", {"Origin": "https://evil.com"}),
        ("Null origin", {"Origin": "null"}),
        ("Trusted subdomain bypass", {"Origin": "https://evil." + urlparse(url).netloc}),
    ]

    table = Table(title=f"[bold green]CORS Check: {url}[/]")
    table.add_column("Test", width=30)
    table.add_column("ACAO Header")
    table.add_column("ACAC Header", width=10)
    table.add_column("Result", width=10)

    results: list[str] = []
    for name, headers in tests:
        try:
            r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            acao = r.headers.get("Access-Control-Allow-Origin", "not set")
            acac = r.headers.get("Access-Control-Allow-Credentials", "false")
            origin_sent = headers["Origin"]

            if acao == origin_sent:
                verdict = "[bold red]FAIL[/]"
            elif acao == "*":
                verdict = "[bold yellow]WARN[/]"
            else:
                verdict = "[bold green]PASS[/]"

            table.add_row(name, acao, acac, verdict)
            results.append(f"{name}: ACAO={acao}  ACAC={acac}")
        except Exception as exc:
            table.add_row(name, f"Error: {exc}", "", "[dim]?[/]")

    console.print(table)

    ts = _ts()
    out = output_dir / f"cors_{ts}.txt"
    out.write_text(f"CORS Check: {url}\n\n" + "\n".join(results))
    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")


# JWT Analyser

def jwt_analyzer(output_dir: Path, ai_manager: Any = None) -> None:
    token = console.input("[bold white][+] Paste JWT token: [/]").strip()
    if not token:
        return

    parts = token.split(".")
    if len(parts) != 3:
        console.print("[red]✗[/] Not a valid JWT (expected 3 parts separated by '.')")
        return

    def _b64decode(s: str) -> dict:
        padding = 4 - len(s) % 4
        s += "=" * padding
        return json.loads(base64.urlsafe_b64decode(s).decode())

    header_dict = payload_dict = {}
    try:
        header_dict = _b64decode(parts[0])
        payload_dict = _b64decode(parts[1])
    except Exception as exc:
        console.print(f"[red]✗[/] Decode error: {exc}")
        return

    table = Table(title="[bold green]JWT Header[/]")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for k, v in header_dict.items():
        table.add_row(str(k), str(v))
    console.print(table)

    table2 = Table(title="[bold green]JWT Payload[/]")
    table2.add_column("Field", style="cyan")
    table2.add_column("Value")
    for k, v in payload_dict.items():
        table2.add_row(str(k), str(v))
    console.print(table2)

    # Security checks
    findings: list[str] = []
    alg = str(header_dict.get("alg", "")).lower()
    if alg == "none":
        findings.append("[bold red]CRITICAL:[/] Algorithm is 'none' — signature verification disabled!")

    # Weak secret check
    try:
        import jwt as pyjwt
        for secret in ("secret", "password", "123456", "", "jwt", "key", "test"):
            try:
                pyjwt.decode(token, secret, algorithms=[header_dict.get("alg", "HS256")])
                findings.append(f"[bold red]CRITICAL:[/] Weak secret found: '{secret}'")
                break
            except Exception:
                pass
    except ImportError:
        findings.append("[dim]PyJWT not installed — weak secret check skipped[/]")

    if findings:
        for f in findings:
            console.print(f"  {f}")
    else:
        console.print("[bold green]✓[/] No obvious vulnerabilities detected")

    if ai_manager and ai_manager.is_available():
        analysis = ai_manager.generate(
            f"Analyse this JWT for security issues:\nHeader: {header_dict}\nPayload: {payload_dict}",
            "You are a web security expert specialising in JWT attacks.",
        )
        if analysis:
            ai_manager.display_analysis(analysis, "JWT Security Analysis")


# Headers Security Audit

_SECURITY_HEADERS = {
    "Strict-Transport-Security": ("HSTS", "max-age=31536000; includeSubDomains"),
    "Content-Security-Policy": ("CSP", "default-src 'self'"),
    "X-Frame-Options": ("X-Frame-Options", "DENY or SAMEORIGIN"),
    "X-Content-Type-Options": ("X-Content-Type-Options", "nosniff"),
    "Referrer-Policy": ("Referrer-Policy", "strict-origin-when-cross-origin"),
    "Permissions-Policy": ("Permissions-Policy", "geolocation=(), microphone=()"),
}


def headers_audit(output_dir: Path, ai_manager: Any = None) -> None:
    url = _ask_url()
    if not url:
        return

    try:
        r = requests.get(url, timeout=10, allow_redirects=True)
        headers = {k.lower(): v for k, v in r.headers.items()}
    except Exception as exc:
        console.print(f"[red]✗[/] Request failed: {exc}")
        return

    table = Table(title=f"[bold green]Header Security Audit: {url}[/]")
    table.add_column("Header", width=30)
    table.add_column("Status", width=15)
    table.add_column("Value / Recommended")

    missing: list[str] = []
    for header, (short, recommended) in _SECURITY_HEADERS.items():
        val = headers.get(header.lower())
        if val:
            table.add_row(header, "[bold green]Present[/]", val[:80])
        else:
            table.add_row(header, "[bold red]Missing[/]", f"[dim]Recommended: {recommended}[/]")
            missing.append(header)

    console.print(table)

    if missing and ai_manager and ai_manager.is_available():
        analysis = ai_manager.generate(
            f"The following security headers are missing on {url}:\n{chr(10).join(missing)}\n\n"
            "For each header, explain: the risk of absence, the recommended value, and an example.",
            "You are a web security expert. Be concise and practical.",
        )
        if analysis:
            ai_manager.display_analysis(analysis, "Header Remediation Advice")

    ts = _ts()
    out = output_dir / f"headers_{ts}.txt"
    out.write_text(f"Headers Audit: {url}\nMissing: {missing}\n")
    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")


# Tech Stack Fingerprinting

def tech_fingerprint(output_dir: Path, ai_manager: Any = None) -> None:
    url = _ask_url()
    if not url:
        return

    fingerprint: dict = {}
    ts = _ts()
    out = str(output_dir / f"fingerprint_{ts}.txt")

    if _check("whatweb"):
        ok, res = run_command(["whatweb", url, "--log-brief", out], timeout=30)
        if ok and res:
            fingerprint["whatweb"] = res[:2000]
            console.print(res[:2000])
    else:
        # Pure-Python fallback
        console.print(f"[yellow]⚠[/] whatweb not installed ([dim]{install_hint('whatweb')}[/]) — falling back to Python")
        try:
            r = requests.get(url, timeout=10, allow_redirects=True)
            server = r.headers.get("Server", "")
            powered = r.headers.get("X-Powered-By", "")
            cookies = str(r.cookies)
            meta_gen = ""
            if b"generator" in r.content.lower():
                import re
                m = re.search(rb'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)', r.content, re.IGNORECASE)
                if m:
                    meta_gen = m.group(1).decode(errors="ignore")

            fingerprint = {"Server": server, "X-Powered-By": powered, "Generator": meta_gen}

            table = Table(title="[bold green]Tech Fingerprint (Python fallback)[/]")
            table.add_column("Field", style="cyan")
            table.add_column("Value")
            for k, v in fingerprint.items():
                if v:
                    table.add_row(k, v)
            console.print(table)

            with open(out, "w") as fh:
                json.dump(fingerprint, fh, indent=2)
        except Exception as exc:
            console.print(f"[red]✗[/] Fingerprinting failed: {exc}")
            return

    if ai_manager and ai_manager.is_available() and fingerprint:
        analysis = ai_manager.generate(
            f"Based on this technology fingerprint of {url}:\n{json.dumps(fingerprint, indent=2)}\n\n"
            "1. List specific CVEs relevant to the detected stack\n"
            "2. Suggest attack vectors\n"
            "3. Recommend next enumeration steps",
            "You are a penetration tester. Be specific and actionable.",
        )
        if analysis:
            ai_manager.display_analysis(analysis, "Stack Analysis & CVE Suggestions")

    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")
