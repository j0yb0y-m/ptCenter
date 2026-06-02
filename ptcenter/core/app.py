"""
ptcenter.core.app
=================
PTCenter main application class and interactive loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from ptcenter.ai.manager import AIManager
from ptcenter.core import config as cfg_module
from ptcenter.core import session as session_module
from ptcenter.core.reporter import generate_report
from ptcenter.ui import menu as ui
from ptcenter.ui.banner import display_banner

# Modules
from ptcenter.modules import (
    exploit,
    network,
    osint,
    password,
    postexploit,
    scanner,
    vuln,
    webapp,
)

load_dotenv()
console = Console()
logger = logging.getLogger(__name__)

# Tool list: (display_name, sysinfo_key)
# Hints are generated at runtime via sysinfo.install_hint() so they match
# the running distro (Arch/pacman vs Debian/apt vs Fedora/dnf).
_TOOL_ENTRIES: list[tuple[str, str]] = [
    ("nmap",        "nmap"),
    ("gobuster",    "gobuster"),
    ("nikto",       "nikto"),
    ("hydra",       "hydra"),
    ("sqlmap",      "sqlmap"),
    ("john",        "john"),
    ("hashcat",     "hashcat"),
    ("ffuf",        "ffuf"),
    ("dalfox",      "dalfox"),
    ("enum4linux",  "enum4linux"),
    ("netexec",     "netexec"),
    ("msfvenom",    "msfvenom"),
    ("sublist3r",   "sublist3r"),
    ("amass",       "amass"),
    ("subfinder",   "subfinder"),
    ("sslscan",     "sslscan"),
    ("crackmapexec","crackmapexec"),
    ("whatweb",     "whatweb"),
    ("exiftool",    "exiftool"),
    ("sherlock",    "sherlock"),
]


def _build_tools_health() -> list[tuple[str, bool, str]]:
    from ptcenter.core.sysinfo import install_hint
    return [
        (name, shutil.which(name) is not None, install_hint(key))
        for name, key in _TOOL_ENTRIES
    ]


def _setup_logging(output_dir: Path) -> None:
    log_file = output_dir / "ptcenter.log"
    handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


class PTCenter:
    """Main application class."""

    def __init__(self, args: Optional[argparse.Namespace] = None) -> None:
        self.config = cfg_module.load_config()
        output_env = os.getenv("OUTPUT_DIR", "")
        self.output_dir = Path(output_env) if output_env else Path(self.config["output_directory"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _setup_logging(self.output_dir)

        self.ai_manager = AIManager()
        self.active_session: Optional[dict] = None
        self.args = args

        try:
            self.default_timeout = int(
                os.getenv("COMMAND_TIMEOUT", str(self.config.get("timeout", 300)))
            )
        except ValueError:
            self.default_timeout = 300

    # Tool health check

    def _tool_health_check(self) -> None:
        tools = _build_tools_health()
        ui.tool_health_table(tools)

    # Session helpers

    def _session_name(self) -> Optional[str]:
        return self.active_session.get("name") if self.active_session else None

    # Menu handlers

    def _scanner_menu(self) -> None:
        while True:
            ui.scanner_menu()
            choice = console.input("[bold white][★] Select: [/]").strip()
            if choice == "1":
                scanner.nmap_scan(self.output_dir, self.active_session, self.ai_manager, self.config.get("auto_ai_analysis", True))
            elif choice == "2":
                scanner.subdomain_scan(self.output_dir, self.active_session, self.ai_manager)
            elif choice == "3":
                scanner.directory_brute_force(self.output_dir, self.active_session)
            elif choice == "4":
                scanner.nikto_scan(self.output_dir, self.active_session, self.ai_manager, self.config.get("auto_ai_analysis", True))
            elif choice == "5":
                scanner.ssl_scan(self.output_dir, self.ai_manager)
            elif choice == "6":
                scanner.dns_enumeration(self.output_dir, self.active_session)
            elif choice == "7":
                scanner.smb_enumeration(self.output_dir, self.active_session)
            elif choice == "8":
                scanner.agentic_recon(self.output_dir, self.ai_manager, self.active_session)
            elif choice == "9":
                break
            else:
                console.print("[red]✗[/] Invalid option")

    def _osint_menu(self) -> None:
        while True:
            ui.osint_menu()
            choice = console.input("[bold white][★] Select: [/]").strip()
            if choice == "1":
                osint.email_intelligence(self.output_dir, self.ai_manager)
            elif choice == "2":
                osint.domain_intelligence(self.output_dir, self.ai_manager)
            elif choice == "3":
                osint.phone_lookup(self.output_dir, self.ai_manager)
            elif choice == "4":
                osint.social_media_search(self.output_dir)
            elif choice == "5":
                osint.metadata_extraction(self.output_dir)
            elif choice == "6":
                osint.whois_lookup(self.output_dir)
            elif choice == "7":
                osint.shodan_search(self.output_dir)
            elif choice == "8":
                break
            else:
                console.print("[red]✗[/] Invalid option")

    def _exploit_menu(self) -> None:
        while True:
            ui.exploit_menu()
            choice = console.input("[bold white][★] Select: [/]").strip()
            if choice == "1":
                exploit.reverse_shell_generator(self.output_dir)
            elif choice == "2":
                exploit.bind_shell_generator(self.output_dir)
            elif choice == "3":
                exploit.msfvenom_generator(self.output_dir)
            elif choice == "4":
                exploit.web_shell_generator(self.output_dir)
            elif choice == "5":
                exploit.sql_injection_payloads(self.output_dir)
            elif choice == "6":
                exploit.xss_payloads(self.output_dir)
            elif choice == "7":
                break
            else:
                console.print("[red]✗[/] Invalid option")

    def _network_menu(self) -> None:
        while True:
            ui.network_menu()
            choice = console.input("[bold white][★] Select: [/]").strip()
            if choice == "1":
                network.arp_spoofing(self.output_dir, self.active_session)
            elif choice == "2":
                network.dns_spoofing(self.output_dir, self.active_session)
            elif choice == "3":
                network.dhcp_starvation(self.output_dir, self.active_session)
            elif choice == "4":
                network.syn_flood(self.output_dir, self.active_session)
            elif choice == "5":
                network.ssl_strip(self.output_dir, self.active_session)
            elif choice == "6":
                network.mitm_setup(self.output_dir, self.active_session)
            elif choice == "7":
                network.network_sniffing(self.output_dir)
            elif choice == "8":
                network.mac_flooding(self.output_dir, self.active_session)
            elif choice == "9":
                break
            else:
                console.print("[red]✗[/] Invalid option")

    def _password_menu(self) -> None:
        while True:
            ui.password_menu()
            choice = console.input("[bold white][★] Select: [/]").strip()
            if choice == "1":
                password.hash_identification(self.output_dir, self.ai_manager)
            elif choice == "2":
                password.hash_cracking(self.output_dir)
            elif choice == "3":
                password.service_brute_force(self.output_dir)
            elif choice == "4":
                password.password_spray(self.output_dir)
            elif choice == "5":
                password.wordlist_manager(self.output_dir)
            elif choice == "6":
                password.ai_hash_analysis(self.output_dir, self.ai_manager)
            elif choice == "7":
                break
            else:
                console.print("[red]✗[/] Invalid option")

    def _webapp_menu(self) -> None:
        while True:
            ui.webapp_menu()
            choice = console.input("[bold white][★] Select: [/]").strip()
            if choice == "1":
                webapp.sqlmap_integration(self.output_dir, self.ai_manager)
            elif choice == "2":
                webapp.parameter_fuzzing(self.output_dir)
            elif choice == "3":
                webapp.xss_scanner(self.output_dir)
            elif choice == "4":
                webapp.cors_check(self.output_dir)
            elif choice == "5":
                webapp.jwt_analyzer(self.output_dir, self.ai_manager)
            elif choice == "6":
                webapp.headers_audit(self.output_dir, self.ai_manager)
            elif choice == "7":
                webapp.tech_fingerprint(self.output_dir, self.ai_manager)
            elif choice == "8":
                break
            else:
                console.print("[red]✗[/] Invalid option")

    def _postexploit_menu(self) -> None:
        while True:
            ui.postexploit_menu()
            choice = console.input("[bold white][★] Select: [/]").strip()
            if choice == "1":
                postexploit.privesc_suggester(self.output_dir, self.ai_manager)
            elif choice == "2":
                postexploit.persistence_reference(self.output_dir)
            elif choice == "3":
                postexploit.lateral_movement(self.output_dir, self.ai_manager)
            elif choice == "4":
                postexploit.data_exfiltration(self.output_dir)
            elif choice == "5":
                postexploit.credential_harvesting(self.output_dir, self.ai_manager)
            elif choice == "6":
                postexploit.lolbins(self.output_dir, self.ai_manager)
            elif choice == "7":
                break
            else:
                console.print("[red]✗[/] Invalid option")

    def _ai_chat_mode(self) -> None:
        """Interactive AI security assistant REPL."""
        if not self.ai_manager.is_available():
            console.print("[red]✗[/] No AI model configured")
            return

        console.print(
            "[bold cyan]🤖 AI Security Assistant[/]  (type 'exit' or 'quit' to leave, 'clear' to reset)\n"
            "[dim]System: Expert penetration tester & CTF player[/]\n"
        )
        self.ai_manager.clear_history()

        while True:
            try:
                user_input = console.input("[bold white]You: [/]").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input.lower() == "clear":
                self.ai_manager.clear_history()
                console.print("[dim]History cleared[/]")
                continue

            response = self.ai_manager.chat(user_input)
            if response:
                console.print("[bold cyan]AI:[/]")
                console.print(Markdown(response))
                console.print()
            else:
                console.print("[red]✗[/] No response from AI")

        console.print("[dim]Exiting chat…[/]")

    def _sessions_menu(self) -> None:
        while True:
            ui.sessions_menu()
            choice = console.input("[bold white][★] Select: [/]").strip()

            if choice == "1":
                # New session
                name = console.input("[bold white][+] Session name (e.g. HTB-Lame): [/]").strip()
                target = console.input("[bold white][+] Target IP/domain: [/]").strip()
                scope_raw = console.input("[bold white][+] Scope CIDRs (comma-sep, Enter to skip): [/]").strip()
                scope = [s.strip() for s in scope_raw.split(",") if s.strip()] if scope_raw else []
                self.active_session = session_module.new_session(name, target, scope)
                console.print(f"[bold green]✓[/] Session created: [cyan]{name}[/]")

            elif choice == "2":
                sessions = session_module.list_sessions()
                if not sessions:
                    console.print("[yellow]⚠[/] No saved sessions")
                    continue
                for idx, s in enumerate(sessions, 1):
                    console.print(f"  [magenta]{idx}[/] {s['name']}  ({s['target']})  findings={s['findings_count']}  {s['created_at'][:10]}")
                raw = console.input("[bold white][+] Load session #: [/]").strip()
                try:
                    idx = int(raw) - 1
                    sid = sessions[idx]["session_id"]
                    loaded = session_module.load_session(sid)
                    if loaded:
                        self.active_session = loaded
                        console.print(f"[bold green]✓[/] Loaded: [cyan]{loaded['name']}[/]")
                    else:
                        console.print("[red]✗[/] Session file not found")
                except (ValueError, IndexError):
                    console.print("[red]✗[/] Invalid selection")

            elif choice == "3":
                if self.active_session:
                    session_module.display_session_summary(self.active_session)
                else:
                    console.print("[yellow]⚠[/] No active session")

            elif choice == "4":
                break
            else:
                console.print("[red]✗[/] Invalid option")

    def _generate_report(self) -> None:
        if not self.active_session:
            console.print("[yellow]⚠[/] No active session — load or create one first")
            return

        tester_name = self.config.get("tester_name", "Unknown")
        console.print(f"[bold blue]🤖[/] Generating executive summary with AI…")

        exec_summary = "No AI model configured."
        recommendations = "No AI model configured."

        if self.ai_manager.is_available():
            findings_text = "\n".join(
                f"[{f.get('severity')}] {f.get('title')}: {f.get('details','')[:200]}"
                for f in self.active_session.get("findings", [])
            ) or "No findings recorded."

            exec_summary = self.ai_manager.generate(
                f"Write a professional 3-paragraph executive summary for this engagement:\n"
                f"Target: {self.active_session.get('target')}\n"
                f"Findings:\n{findings_text}",
                "You are a senior penetration testing consultant writing for a non-technical audience.",
            ) or exec_summary

            recommendations = self.ai_manager.generate(
                f"Generate a prioritised remediation checklist for these findings:\n{findings_text}",
                "You are a security consultant. Be specific, actionable, and prioritise by risk.",
            ) or recommendations

        report_path = generate_report(
            session=self.active_session,
            output_dir=self.output_dir,
            tester_name=tester_name,
            executive_summary=exec_summary,
            recommendations=recommendations,
            open_browser=True,
        )
        console.print(f"[bold green]✓[/] Report: [cyan]{report_path}[/]")

    def _settings_menu(self) -> None:
        while True:
            loaded = list(self.ai_manager.get_available_models().keys())
            ui.settings_menu(
                output_dir=str(self.output_dir),
                ai_status="[bold green]Enabled[/]" if self.ai_manager.is_available() else "[bold red]Disabled[/]",
                ai_model=self.ai_manager.active_model_name(),
                loaded_models=", ".join(loaded) or "None",
                auto_analysis=bool(self.config.get("auto_ai_analysis", True)),
            )
            choice = console.input("[bold white][★] Select: [/]").strip()

            if choice == "1":
                # Select AI model
                available = self.ai_manager.get_available_models()
                if not available:
                    console.print("[yellow]⚠[/] No models loaded — add API keys to .env")
                    continue
                keys = list(available.keys())
                for idx, key in enumerate(keys, 1):
                    m = available[key]
                    active = " [bold green]◄ ACTIVE[/]" if m is self.ai_manager.active_model else ""
                    console.print(f"  [magenta]{idx}[/] {m.display_name}{active}")
                raw = console.input(f"[bold white][+] Select [1-{len(keys)}]: [/]").strip()
                try:
                    idx = int(raw) - 1
                    if 0 <= idx < len(keys):
                        self.ai_manager.select_model(keys[idx])
                        console.print(f"[bold green]✓[/] Switched to: {self.ai_manager.active_model_name()}")
                except ValueError:
                    console.print("[red]✗[/] Invalid input")

            elif choice == "2":
                self.config["auto_ai_analysis"] = not self.config.get("auto_ai_analysis", True)
                status = "enabled" if self.config["auto_ai_analysis"] else "disabled"
                cfg_module.save_config(self.config)
                console.print(f"[bold green]✓[/] Auto AI analysis {status}")

            elif choice == "3":
                self._show_api_help()

            elif choice == "4":
                confirm = console.input(f"[bold yellow]Clear all files in {self.output_dir}? (yes/no): [/]").strip()
                if confirm == "yes":
                    shutil.rmtree(self.output_dir)
                    self.output_dir.mkdir()
                    console.print("[bold green]✓[/] Cleared")

            elif choice == "5":
                log_file = self.output_dir / "ptcenter.log"
                if log_file.exists():
                    lines = log_file.read_text(errors="replace").splitlines()
                    console.print("\n".join(lines[-50:]))
                else:
                    console.print("[yellow]⚠[/] No log file yet")

            elif choice == "6":
                break
            else:
                console.print("[red]✗[/] Invalid option")

    def _show_api_help(self) -> None:
        console.print("""
[bold green]① Google Gemini — 100% FREE[/]
  Key:   GEMINI_API_KEY
  Get:   https://aistudio.google.com/app/apikey
  Limit: 15 req/min · 1M tokens/day · no credit card

[bold blue]② OpenAI GPT-4o — Paid[/]
  Key:   OPENAI_API_KEY
  Model: OPENAI_MODEL (default: gpt-4o)
  Get:   https://platform.openai.com/api-keys

[bold magenta]③ Anthropic Claude — Paid[/]
  Key:   ANTHROPIC_API_KEY
  Model: CLAUDE_MODEL (default: claude-3-5-haiku-latest)
  Get:   https://console.anthropic.com/

[bold yellow]④ Ollama Local — 100% FREE & Offline[/]
  Host:  OLLAMA_HOST (default: http://localhost:11434)
  Model: OLLAMA_MODEL (default: llama3)
  Install: https://ollama.com/download
  Pull:  ollama pull llama3

[dim]Set ACTIVE_AI_MODEL=gemini|openai|claude|ollama in .env for default[/]
[bold yellow]⚠[/] Never commit your .env to version control!
""")

    # CLI / non-interactive mode

    def _run_cli(self) -> None:
        """Non-interactive execution via argparse args."""
        args = self.args
        if not args:
            return
        module = getattr(args, "module", None)
        tool = getattr(args, "tool", None)
        target = getattr(args, "target", None)

        if module == "scanner" and tool == "nmap" and target:
            from ptcenter.core.validator import validate_target
            if not validate_target(target):
                console.print(f"[red]✗[/] Invalid target: {target}")
                sys.exit(1)
            scanner.nmap_scan(self.output_dir, None, self.ai_manager, True)

        elif module == "vuln" and target:
            vuln.vulnerability_info(self.output_dir, self.ai_manager)
        else:
            console.print("[yellow]⚠[/] CLI mode: unsupported module/tool combination — launching interactive mode")
            self._interactive()

    # Interactive loop

    def _interactive(self) -> None:
        display_banner(
            ai_model_name=self.ai_manager.active_model_name(),
            ai_available=self.ai_manager.is_available(),
            model_count=len(self.ai_manager.get_available_models()),
            output_dir=self.output_dir,
        )

        no_check = getattr(self.args, "no_check", False) if self.args else False
        if not no_check:
            self._tool_health_check()

        if not self.ai_manager.is_available():
            console.print(
                "\n[bold yellow]💡[/] No AI model configured.  "
                "See Settings → View API Setup Help\n"
            )

        while True:
            ui.main_menu(session_name=self._session_name())
            choice = console.input("\n[bold white][★] Select option: [/]").strip()

            if choice == "1":
                self._scanner_menu()
            elif choice == "2":
                self._osint_menu()
            elif choice == "3":
                vuln.vulnerability_info(self.output_dir, self.ai_manager)
            elif choice == "4":
                self._exploit_menu()
            elif choice == "5":
                self._network_menu()
            elif choice == "6":
                self._password_menu()
            elif choice == "7":
                self._webapp_menu()
            elif choice == "8":
                self._postexploit_menu()
            elif choice == "9":
                self._ai_chat_mode()
            elif choice == "10":
                self._sessions_menu()
            elif choice == "11":
                self._generate_report()
            elif choice == "12":
                self._settings_menu()
            elif choice == "13":
                console.print("\n[bold green]✓[/] Thank you for using ptCenter!")
                console.print("[bold yellow]⚠[/] Only test systems you have permission to test!\n")
                break
            else:
                console.print("[red]✗[/] Invalid option — select 1-13")

    # Entry point

    def run(self) -> None:
        if self.args and getattr(self.args, "module", None):
            self._run_cli()
        else:
            self._interactive()
