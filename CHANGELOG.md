# Changelog

All notable changes to ptCenter are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] — 2026-06-01

### Added
- Full architectural refactor from monolithic script to structured Python
  package (`ptcenter/` with `core/`, `modules/`, `ai/`, `ui/` sub-packages).
- **AI integration** — multi-model manager supporting Google Gemini,
  OpenAI, Anthropic Claude, and local Ollama simultaneously with runtime
  switching.
- **Agentic recon** — AI-driven autonomous scan loop that decides and
  executes the next best action based on prior scan results.
- **Chunked analysis** — automatic splitting of large scan outputs (>7,500
  chars) into summarized chunks with a final synthesis pass.
- **Engagement sessions** — persistent JSON-backed sessions tracking target,
  scope (CIDR-aware), findings, and completed scans across reboots.
- **HTML report generation** — AI-written executive summary and remediation
  checklist, fully self-contained (no CDN), opens in browser.
- **Password attacks module** — hash identification, hashcat/john cracking,
  Hydra service brute force, password spray, wordlist manager.
- **Web application testing module** — SQLMap, ffuf parameter fuzzing,
  dalfox XSS scanner, CORS check, JWT analyzer, security headers audit,
  WhatWeb fingerprinting.
- **Post-exploitation module** — PrivEsc suggester (Linux/Windows),
  persistence reference, lateral movement, data exfiltration, LOLBins.
- **`core.runner`** — centralized `shell=False` subprocess wrapper
  (`run_command`, `stream_command`) used by every module.
- **`core.validator`** — input validation for IP, CIDR, domain, URL, port,
  and nmap flags. Shell injection prevention via sanitized flag lists.
- **`core.sysinfo`** — distro detection (Arch, Debian, Fedora families) with
  per-distro install hints rendered at runtime.
- **`core.session`** — UUID-keyed JSON sessions with finding severity
  tracking and scope enforcement.
- **`core.reporter`** — Jinja2 HTML report with dark theme, severity badges,
  findings cards, and AI-generated content sections.
- **`ui.menu`** — full Rich-rendered interactive menu system.
- **`ui.banner`** — distro-aware ASCII art banner with live AI status.
- Arch/CachyOS/BlackArch automated installer (`install_arch.sh`) supporting
  `--no-blackarch` and `--aur-helper` flags.
- `pyproject.toml` with optional dependency extras: `ai`, `osint`, `network`,
  `full`. Built with `hatchling`, managed with `uv`.
- Non-interactive CLI mode (`--module`, `--tool`, `--target`, `--output`,
  `--no-check`, `--version`).
- Rotating file logger (`ptcenter.log`, max 5MB, 3 backups).

### Changed
- All subprocess calls migrated from `shell=True` string concatenation to
  `shell=False` list-based invocation.
- Nmap output now parsed from XML for structured AI analysis.
- Subdomain scanner tries `subfinder` → `amass` → `sublist3r` in order.
- Directory brute forcer tries `gobuster` → `ffuf` → `dirb` → `dirsearch`.

### Removed
- Legacy monolithic single-file script (v1.x).

---

## [1.x] — Earlier (pre-refactor)

Single-file monolithic script with basic menu, no session management,
no input validation, and `shell=True` subprocess calls.
Not maintained. Upgrade to v2.
