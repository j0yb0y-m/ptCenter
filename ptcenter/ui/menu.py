"""ptcenter.ui.menu — menu rendering helpers."""

from __future__ import annotations

import shutil

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _width() -> int:
    return shutil.get_terminal_size().columns


def print_separator() -> None:
    console.print("[bold cyan]" + "═" * min(_width(), 78) + "[/]")


def print_sub_separator() -> None:
    console.print("[dim cyan]" + "─" * min(_width(), 78) + "[/]")


def main_menu(session_name: str | None = None) -> None:
    session_line = f"[dim]Active session:[/] [cyan]{session_name}[/]" if session_name else "[dim]No active session[/]"
    console.print(
        Panel(
            Text.from_markup(
                f"\n"
                f"  [bold magenta] 1[/]  ► [bold green]Network Scanning[/]      - Port, service & vulnerability scanning\n"
                f"  [bold magenta] 2[/]  ► [bold blue]OSINT Investigation[/]   - Open source intelligence gathering\n"
                f"  [bold magenta] 3[/]  ► [bold yellow]Vulnerability Info[/]    - CVE lookup and AI security analysis\n"
                f"  [bold magenta] 4[/]  ► [bold red]Exploit Development[/]   - Shells, payloads & web shells\n"
                f"  [bold magenta] 5[/]  ► [bold cyan]Network Attacks[/]       - ARP, DNS, MITM & DoS attacks\n"
                f"  [bold magenta] 6[/]  ► [bold magenta]Password Attacks[/]      - Hash cracking, brute force, spray\n"
                f"  [bold magenta] 7[/]  ► [bold green]Web App Testing[/]       - SQLi, XSS, CORS, JWT, headers\n"
                f"  [bold magenta] 8[/]  ► [bold yellow]Post-Exploitation[/]     - Privesc, persistence, lateral move\n"
                f"  [bold magenta] 9[/]  ► [bold blue]AI Security Chat[/]      - Interactive AI assistant\n"
                f"  [bold magenta]10[/]  ► [bold cyan]Sessions[/]              - New / load / view session\n"
                f"  [bold magenta]11[/]  ► [bold green]Generate Report[/]       - HTML engagement report\n"
                f"  [bold magenta]12[/]  ► [bold white]Settings[/]              - Configure AI models & options\n"
                f"  [bold magenta]13[/]  ► [bold red]Exit[/]\n\n"
                f"  {session_line}\n"
            ),
            title="[bold cyan]MAIN MENU[/]",
            border_style="cyan",
        )
    )


def scanner_menu() -> None:
    console.print(
        Panel(
            Text.from_markup(
                f"\n"
                f"  [bold magenta]1[/]  - Nmap Port Scan\n"
                f"  [bold magenta]2[/]  - Subdomain Discovery (Sublist3r / Amass / Subfinder)\n"
                f"  [bold magenta]3[/]  - Directory Brute Force (Gobuster / Dirb / Dirsearch)\n"
                f"  [bold magenta]4[/]  - Web Application Scan (Nikto)\n"
                f"  [bold magenta]5[/]  - SSL/TLS Analysis (SSLScan / testssl.sh / Python)\n"
                f"  [bold magenta]6[/]  - DNS Enumeration\n"
                f"  [bold magenta]7[/]  - SMB Enumeration (netexec / enum4linux)\n"
                f"  [bold magenta]8[/]  - Agentic Recon (AI-driven attack chain)\n"
                f"  [bold magenta]9[/]  - Back to Main Menu\n"
            ),
            title="[bold cyan]🔍 SCANNING MODULE[/]",
            border_style="cyan",
        )
    )


def osint_menu() -> None:
    console.print(
        Panel(
            Text.from_markup(
                f"\n"
                f"  [bold magenta]1[/] - Email / Username Intelligence\n"
                f"  [bold magenta]2[/] - Domain / IP Intelligence\n"
                f"  [bold magenta]3[/] - Phone Number Lookup\n"
                f"  [bold magenta]4[/] - Social Media Search\n"
                f"  [bold magenta]5[/] - Metadata Extraction (ExifTool)\n"
                f"  [bold magenta]6[/] - WHOIS Lookup\n"
                f"  [bold magenta]7[/] - Shodan Search\n"
                f"  [bold magenta]8[/] - Back to Main Menu\n"
            ),
            title="[bold cyan]🔍 OSINT MODULE[/]",
            border_style="cyan",
        )
    )


def exploit_menu() -> None:
    console.print(
        Panel(
            Text.from_markup(
                f"\n"
                f"  [bold magenta]1[/] - Reverse Shell Generator\n"
                f"  [bold magenta]2[/] - Bind Shell Generator\n"
                f"  [bold magenta]3[/] - Msfvenom Payload Generator\n"
                f"  [bold magenta]4[/] - Web Shell Generator\n"
                f"  [bold magenta]5[/] - SQL Injection Payloads\n"
                f"  [bold magenta]6[/] - XSS Payloads\n"
                f"  [bold magenta]7[/] - Back to Main Menu\n"
            ),
            title="[bold red]💉 EXPLOIT DEVELOPMENT[/]",
            border_style="red",
        )
    )


def network_menu() -> None:
    console.print(
        Panel(
            Text.from_markup(
                f"\n"
                f"  [bold yellow]⚠[/] These tools can disrupt networks. Use only on authorised systems!\n\n"
                f"  [bold magenta]1[/] - ARP Spoofing / Poisoning\n"
                f"  [bold magenta]2[/] - DNS Spoofing\n"
                f"  [bold magenta]3[/] - DHCP Starvation\n"
                f"  [bold magenta]4[/] - SYN Flood (DoS)\n"
                f"  [bold magenta]5[/] - SSL Strip Attack\n"
                f"  [bold magenta]6[/] - Man-in-the-Middle Setup\n"
                f"  [bold magenta]7[/] - Network Sniffing\n"
                f"  [bold magenta]8[/] - MAC Flooding\n"
                f"  [bold magenta]9[/] - Back to Main Menu\n"
            ),
            title="[bold yellow]⚡ NETWORK ATTACKS[/]",
            border_style="yellow",
        )
    )


def password_menu() -> None:
    console.print(
        Panel(
            Text.from_markup(
                f"\n"
                f"  [bold magenta]1[/] - Hash Identification\n"
                f"  [bold magenta]2[/] - Hash Cracking (hashcat / john)\n"
                f"  [bold magenta]3[/] - Service Brute Force (hydra)\n"
                f"  [bold magenta]4[/] - Password Spray\n"
                f"  [bold magenta]5[/] - Wordlist Manager\n"
                f"  [bold magenta]6[/] - AI Hash Analysis\n"
                f"  [bold magenta]7[/] - Back to Main Menu\n"
            ),
            title="[bold magenta]🔑 PASSWORD ATTACKS[/]",
            border_style="magenta",
        )
    )


def webapp_menu() -> None:
    console.print(
        Panel(
            Text.from_markup(
                f"\n"
                f"  [bold magenta]1[/] - SQLMap Integration\n"
                f"  [bold magenta]2[/] - Parameter Fuzzing (ffuf)\n"
                f"  [bold magenta]3[/] - XSS Scanner (dalfox)\n"
                f"  [bold magenta]4[/] - CORS Misconfiguration Check\n"
                f"  [bold magenta]5[/] - JWT Analyser\n"
                f"  [bold magenta]6[/] - Headers Security Audit\n"
                f"  [bold magenta]7[/] - Tech Stack Fingerprinting\n"
                f"  [bold magenta]8[/] - Back to Main Menu\n"
            ),
            title="[bold green]🌐 WEB APP TESTING[/]",
            border_style="green",
        )
    )


def postexploit_menu() -> None:
    console.print(
        Panel(
            Text.from_markup(
                f"\n"
                f"  [dim]Reference / educational content — run commands manually after getting a shell[/]\n\n"
                f"  [bold magenta]1[/] - Privilege Escalation Suggester\n"
                f"  [bold magenta]2[/] - Persistence Techniques Reference\n"
                f"  [bold magenta]3[/] - Lateral Movement Checklist\n"
                f"  [bold magenta]4[/] - Data Exfiltration Techniques\n"
                f"  [bold magenta]5[/] - Credential Harvesting Reference\n"
                f"  [bold magenta]6[/] - Living-Off-The-Land Binaries (GTFOBins / LOLBAS)\n"
                f"  [bold magenta]7[/] - Back to Main Menu\n"
            ),
            title="[bold yellow]🎯 POST-EXPLOITATION[/]",
            border_style="yellow",
        )
    )


def sessions_menu() -> None:
    console.print(
        Panel(
            Text.from_markup(
                f"\n"
                f"  [bold magenta]1[/] - New Session\n"
                f"  [bold magenta]2[/] - Load Session\n"
                f"  [bold magenta]3[/] - View Current Session\n"
                f"  [bold magenta]4[/] - Back to Main Menu\n"
            ),
            title="[bold cyan]📁 SESSIONS[/]",
            border_style="cyan",
        )
    )


def settings_menu(
    output_dir: str,
    ai_status: str,
    ai_model: str,
    loaded_models: str,
    auto_analysis: bool,
) -> None:
    console.print(
        Panel(
            Text.from_markup(
                f"\n"
                f"  [dim]Output Directory :[/] [cyan]{output_dir}[/]\n"
                f"  [dim]AI Status        :[/] {ai_status}\n"
                f"  [dim]Active AI Model  :[/] [cyan]{ai_model}[/]\n"
                f"  [dim]Available Models :[/] [cyan]{loaded_models}[/]\n"
                f"  [dim]Auto AI Analysis :[/] {'[green]On[/]' if auto_analysis else '[red]Off[/]'}\n\n"
                f"  [bold magenta]1[/] - Change Active AI Model\n"
                f"  [bold magenta]2[/] - Toggle Auto AI Analysis\n"
                f"  [bold magenta]3[/] - View API Setup Help\n"
                f"  [bold magenta]4[/] - Clear Output Directory\n"
                f"  [bold magenta]5[/] - View Logs\n"
                f"  [bold magenta]6[/] - Back to Main Menu\n"
            ),
            title="[bold white]⚙️  SETTINGS[/]",
            border_style="white",
        )
    )


def tool_health_table(tools: list[tuple[str, bool, str]]) -> None:
    """Display a Rich table of tool health status."""
    table = Table(title="[bold cyan]Tool Health Check[/]", show_lines=False)
    table.add_column("Tool", style="bold white", width=20)
    table.add_column("Status", width=14)
    table.add_column("Install Hint")

    for tool_name, installed, hint in tools:
        status = "[bold green]✓ installed[/]" if installed else "[bold red]✗ missing[/]"
        table.add_row(tool_name, status, f"[dim]{hint}[/]" if not installed else "")

    console.print(table)
