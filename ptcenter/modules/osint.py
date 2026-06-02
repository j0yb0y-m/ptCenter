"""
ptcenter.modules.osint
======================
All OSINT functions: email/username, domain/IP, phone, social media,
metadata, WHOIS, Shodan.  All subprocess calls via core.runner (shell=False).
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from ptcenter.core.sysinfo import install_hint
from ptcenter.core.runner import run_command
from ptcenter.core.validator import validate_domain, validate_target

console = Console()


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _check(tool: str) -> bool:
    return shutil.which(tool) is not None


# Email / Username Intelligence


def email_intelligence(output_dir: Path, ai_manager: Any = None) -> None:
    query = console.input("[bold white][+] Email / username: [/]").strip()
    if not query:
        console.print("[red]✗[/] Input cannot be empty")
        return

    ts = _ts()
    out = output_dir / f"osint_email_{ts}.txt"
    lines: list[str] = [
        f"Email/Username OSINT Report\nTarget: {query}\nDate: {datetime.now()}\n{'='*60}\n"
    ]

    if "@" in query:
        username, domain = query.split("@", 1)
        lines.append(f"Username : {username}\nDomain   : {domain}")
        email_re = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
        lines.append("✓ Valid email format" if email_re.match(query) else "✗ Invalid format")

        # Holehe CLI
        if _check("holehe"):
            console.print("[bold blue]▶[/] Running holehe…")
            ok, res = run_command(["holehe", query], timeout=120)
            if ok and res:
                lines.append(f"\n=== Holehe Account Check ===\n{res}")
                console.print(res)
        else:
            console.print(f"[yellow]⚠[/] holehe not installed — [dim]{install_hint('holehe')}[/]")
            # Programmatic fallback
            try:
                import trio
                import httpx as _httpx
                from holehe.core import get_functions  # type: ignore

                async def _run_holehe(email: str):
                    results = []
                    functions = get_functions()
                    async with _httpx.AsyncClient() as client:
                        for func in functions:
                            try:
                                out_list: list = []
                                await func(email, client, out_list)
                                results.extend(out_list)
                            except Exception:
                                pass
                    return results

                console.print("[bold blue]▶[/] Running holehe Python API…")
                holehe_results = trio.run(_run_holehe, query)
                found = [r for r in holehe_results if r.get("exists")]
                if found:
                    found_lines = "\n".join(f"  ✓ {r['name']}" for r in found)
                    lines.append(f"\n=== Holehe (API) ===\n{found_lines}")
                    console.print(found_lines)
            except ImportError:
                pass
            except Exception as exc:
                console.print(f"[yellow]⚠[/] Holehe API: {exc}")

    # AI OSINT guidance
    if ai_manager and ai_manager.is_available():
        prompt = (
            f"Provide OSINT intelligence guidance for: {query} including: \n\n"
            "1. Public information sources and potential data repositories \n"
            "2. Common platforms, networks, and websites where this identifier may be visible\n"
            "3. Relevant breach databases and dark web marketplaces to search\n"
            "4. Recommended tools and software for efficient OSINT gathering and analysis\n"
            "5. Key legal and ethical considerations for conducting this type of research"
        )
        analysis = ai_manager.generate(prompt, "You are an OSINT expert. Provide practical, ethical guidance.")
        if analysis:
            lines.append(f"\n=== AI OSINT Analysis ===\n{analysis}")
            ai_manager.display_analysis(analysis, "OSINT Guidance")

    with open(out, "w") as fh:
        fh.write("\n".join(lines))

    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")
    console.print("\n[bold blue]Recommended tools:[/]")
    console.print("  Sherlock  : [dim]sherlock <username>[/]")
    console.print("  Holehe    : [dim]holehe <email>[/]")
    console.print("  HIBP      : https://haveibeenpwned.com/")
    console.print("  Hunter.io : https://hunter.io/")


# Domain / IP Intelligence

def domain_intelligence(output_dir: Path, ai_manager: Any = None) -> None:
    target = console.input("[bold white][+] Domain / IP: [/]").strip()
    if not target or not validate_target(target):
        console.print("[red]✗[/] Invalid target")
        return

    ts = _ts()
    out = output_dir / f"osint_domain_{ts}.txt"
    lines: list[str] = [f"Domain/IP Intelligence: {target}\n{datetime.now()}\n{'='*60}\n"]

    # DNS
    ok, res = run_command(["dig", target, "+short"], timeout=10)
    if ok and res:
        lines.append(f"=== DNS Resolution ===\n{res}")

    # WHOIS
    if _check("whois"):
        ok, res = run_command(["whois", target], timeout=20)
        if ok:
            lines.append(f"\n=== WHOIS ===\n{res[:3000]}")

    # Reverse DNS
    ok, res = run_command(["dig", "-x", target, "+short"], timeout=10)
    if ok and res:
        lines.append(f"\n=== Reverse DNS ===\n{res}")

    with open(out, "w") as fh:
        fh.write("\n".join(lines))

    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")



# Phone Number Lookup

def phone_lookup(output_dir: Path, ai_manager: Any = None) -> None:
    phone = console.input("[bold white][+] Phone number (E.164, e.g. +9647701234567): [/]").strip()
    if not phone:
        return

    console.print(f"\n[bold blue]Phone:[/] {phone}")

    # phonenumbers library (proper parsing)
    try:
        import phonenumbers
        from phonenumbers import geocoder, carrier

        parsed = phonenumbers.parse(phone)
        if phonenumbers.is_valid_number(parsed):
            table = Table(title="[bold green]Phone Info[/]")
            table.add_column("Field", style="cyan")
            table.add_column("Value")
            table.add_row("Number (E.164)", phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164))
            table.add_row("Country Code", str(parsed.country_code))
            table.add_row("National", str(parsed.national_number))
            table.add_row("Region", geocoder.description_for_number(parsed, "en"))
            table.add_row("Carrier", carrier.name_for_number(parsed, "en"))
            console.print(table)
        else:
            console.print("[yellow]⚠[/] Number appears invalid per phonenumbers library")
    except ImportError:
        console.print(f"[yellow]⚠[/] phonenumbers not installed — [dim]uv add phonenumbers[/]")

    # TrueCaller CLI
    if _check("truecallerpy"):
        run_command(["truecallerpy", "login"], timeout=30)
        run_command(["truecallerpy", "-i", phone], timeout=30)
    else:
        installation_id = __import__("os").getenv("TRUECALLER_INSTALLATION_ID", "")
        if installation_id:
            try:
                import asyncio
                from truecallerpy import search_phonenumber  # type: ignore
                import phonenumbers as _pn
                parsed2 = _pn.parse(phone)
                region = _pn.region_code_for_number(parsed2) or "IQ"
                result = asyncio.run(search_phonenumber(phone, region, installation_id))
                if result and result.get("data"):
                    d = result["data"]
                    console.print(f"[bold green]TrueCaller: {d.get('name','N/A')} / {d.get('carrier','N/A')}[/]")
            except Exception as exc:
                console.print(f"[yellow]⚠[/] TrueCaller API: {exc}")

    console.print("\n[bold blue]Resources:[/]")
    console.print("  TrueCaller : https://www.truecaller.com/")
    console.print("  PhoneInfoga: https://github.com/sundowndev/phoneinfoga")


# Social Media Search

def social_media_search(output_dir: Path) -> None:
    username = console.input("[bold white][+] Username: [/]").strip()
    if not username:
        return

    platforms = [
        ("Twitter/X", f"https://twitter.com/{username}"),
        ("Instagram", f"https://instagram.com/{username}"),
        ("GitHub", f"https://github.com/{username}"),
        ("LinkedIn", f"https://linkedin.com/in/{username}"),
        ("Facebook", f"https://facebook.com/{username}"),
        ("Reddit", f"https://reddit.com/user/{username}"),
        ("TikTok", f"https://tiktok.com/@{username}"),
        ("YouTube", f"https://youtube.com/@{username}"),
    ]

    table = Table(title=f"[bold green]Profile URLs for '{username}'[/]")
    table.add_column("Platform", style="cyan")
    table.add_column("URL")
    for platform, url in platforms:
        table.add_row(platform, url)
    console.print(table)

    ts = _ts()
    out = output_dir / f"social_{username}_{ts}.txt"

    if _check("sherlock"):
        console.print("[bold blue]▶[/] Running Sherlock…")
        ok, _ = run_command(["sherlock", username, "--output", str(out), "--print-found"], timeout=180)
        if ok:
            console.print(f"[bold green]✓[/] Sherlock results: [cyan]{out}[/]")
    else:
        console.print(f"[yellow]⚠[/] Sherlock not installed — [dim]{install_hint('sherlock')}[/]")

    if _check("maigret"):
        console.print("[bold blue]▶[/] Running Maigret…")
        maigret_dir = output_dir / f"maigret_{username}_{ts}"
        maigret_dir.mkdir(exist_ok=True)
        run_command(
            ["maigret", username, "--no-recursion", "-n", "100",
             "--folderoutput", str(maigret_dir), "--print-found"],
            timeout=180,
        )
        console.print(f"[bold green]✓[/] Maigret results: [cyan]{maigret_dir}[/]")
    else:
        console.print(f"[yellow]⚠[/] Maigret not installed — [dim]{install_hint('maigret')}[/]")


# Metadata Extraction

def metadata_extraction(output_dir: Path) -> None:
    if not _check("exiftool"):
        console.print(f"[red]✗[/] exiftool not installed.  [dim]{install_hint('exiftool')}[/]")
        return

    file_path = console.input("[bold white][+] File path: [/]").strip()
    if not file_path or not Path(file_path).exists():
        console.print("[red]✗[/] File not found")
        return

    ts = _ts()
    out = str(output_dir / f"metadata_{ts}.txt")
    run_command(["exiftool", file_path], out, timeout=30)
    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")


# WHOIS Lookup

def whois_lookup(output_dir: Path) -> None:
    target = console.input("[bold white][+] Domain / IP: [/]").strip()
    if not target:
        return

    ts = _ts()
    out = str(output_dir / f"whois_{target}_{ts}.txt")
    run_command(["whois", target], out, timeout=30)
    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")


# Shodan Search

def shodan_search(output_dir: Path) -> None:
    if not _check("shodan"):
        console.print(f"[red]✗[/] Shodan CLI not installed.  [dim]{install_hint('shodan')}[/]")
        console.print("[dim]Then: shodan init YOUR_API_KEY[/]")
        return

    query = console.input("[bold white][+] Shodan query: [/]").strip()
    if not query:
        return

    ts = _ts()
    out = str(output_dir / f"shodan_{ts}.txt")
    # shlex.split is safe here — we validate / sanitize the query string
    # We pass each word as a separate list element to avoid shell injection
    run_command(["shodan", "search"] + query.split(), out, timeout=60)
    console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")
