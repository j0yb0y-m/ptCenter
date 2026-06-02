

```
 ██████╗ ████████╗    ██████╗███████╗███╗   ██╗████████╗███████╗██████╗
 ██╔══██╗╚══██╔══╝   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗
 ██████╔╝   ██║      ██║     █████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝
 ██╔═══╝    ██║      ██║     ██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗
 ██║        ██║      ╚██████╗███████╗██║ ╚████║   ██║   ███████╗██║  ██║
 ╚═╝        ╚═╝       ╚═════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
```

**Advanced Python Penetration Testing Toolkit with AI Integration**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-red?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey?style=flat-square&logo=linux)](https://kernel.org)
[![Version](https://img.shields.io/badge/Version-2.0.0-cyan?style=flat-square)](https://github.com/j0yb0y-m/ptcenter)
[![Modules](https://img.shields.io/badge/Modules-8-green?style=flat-square)]()
[![AI Models](https://img.shields.io/badge/AI%20Models-4-purple?style=flat-square)]()

*Built for security professionals, CTF players, and red teamers.*

---

> ⚠️ **Legal Disclaimer** — ptCenter is designed exclusively for authorized penetration testing, security research, and CTF competitions. You are solely responsible for ensuring you have written permission to test any system. Unauthorized use of this tool against systems you do not own or have explicit permission to test is illegal and unethical. The author assumes no liability for misuse.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Modules](#modules)
- [AI Integration](#ai-integration)
- [Session Management](#session-management)
- [Reporting](#reporting)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

**ptCenter v2** is a modular, AI-augmented penetration testing toolkit written in Python. It unifies the most essential offensive security workflows — from reconnaissance to post-exploitation — under a single, Rich-powered terminal interface.

What separates ptCenter from a simple script collection:

- **AI-native design** — every module can invoke an AI model (Gemini, OpenAI, Claude, or local Ollama) to analyze scan output, identify vulnerabilities, and suggest next steps automatically.
- **Agentic recon** — the AI decides the next scan action based on what it has seen so far, enabling hands-off automated reconnaissance chains.
- **Engagement sessions** — persistent, JSON-backed sessions track your target, scope, findings, and completed scans across reboots.
- **Distro-aware** — runtime install hints are served for Arch/CachyOS, Debian/Kali/Parrot, and Fedora-family systems.

---

## Features

| Category | Capability |
|---|---|
| **Scanning** | nmap (10 profiles), subdomain enumeration, directory brute force, Nikto, SSL/TLS scan, DNS enumeration, SMB enumeration, AI-driven agentic recon |
| **OSINT** | Email & username intelligence (Holehe), domain & IP intel, phone lookup, social media search (Sherlock/Maigret), metadata extraction (ExifTool), WHOIS, Shodan |
| **Vulnerability** | CVE/vulnerability lookup with AI-powered analysis |
| **Exploitation** | Reverse shell generator (13 languages), bind shells, msfvenom payload builder, web shell generator, SQLi payloads, XSS payloads |
| **Network Attacks** | ARP spoofing, DNS spoofing, DHCP starvation, SYN flood, SSL stripping, MITM setup, network sniffing, MAC flooding |
| **Password Attacks** | Hash identification (AI fallback), hash cracking (hashcat/john), service brute force (hydra), password spraying, wordlist manager |
| **Web App Testing** | SQLMap integration, parameter fuzzing (ffuf), XSS scanner (dalfox), CORS checker, JWT analyzer, security headers audit, tech fingerprinting (WhatWeb) |
| **Post-Exploitation** | PrivEsc suggester (Linux/Windows), persistence reference, lateral movement reference, data exfiltration helper, credential harvesting, LOLBins reference |
| **AI Features** | Scan analysis, chunked analysis for large outputs, interactive security chat, agentic recon, AI-generated HTML reports |
| **Sessions** | Create/load/save engagement sessions, scope enforcement (CIDR-aware), finding tracking |
| **Reporting** | AI-generated HTML report with executive summary, risk matrix, findings cards, and remediation checklist |

---

## Architecture

```
ptcenter/
├── __main__.py          # Entry point & CLI argument parser
├── core/
│   ├── app.py           # Main application class & interactive loop
│   ├── config.py        # Config load/save (~/.ptcenter_config.json)
│   ├── reporter.py      # Jinja2 HTML report generation
│   ├── runner.py        # Safe subprocess wrapper (shell=False always)
│   ├── session.py       # Engagement session persistence
│   ├── sysinfo.py       # Distro detection & install hints
│   └── validator.py     # Input validation (IP, domain, URL, port)
├── ai/
│   ├── manager.py       # Multi-model manager, chunked analysis, chat history
│   └── models/
│       ├── base.py      # Abstract base model
│       ├── gemini.py    # Google Gemini (gemini-2.0-flash)
│       ├── openai.py    # OpenAI GPT (gpt-4o default)
│       ├── claude.py    # Anthropic Claude (claude-3-5-haiku-latest default)
│       └── ollama.py    # Local Ollama (any model)
├── modules/
│   ├── scanner.py       # Scanning & enumeration
│   ├── osint.py         # Open-source intelligence
│   ├── vuln.py          # Vulnerability research
│   ├── exploit.py       # Payload & shell generation
│   ├── network.py       # Network attacks
│   ├── password.py      # Password attacks
│   ├── webapp.py        # Web application testing
│   └── postexploit.py   # Post-exploitation reference
└── ui/
    ├── banner.py        # ASCII art banner
    ├── menu.py          # All Rich-rendered menus
    └── colors.py        # Color constants
```

---

## Installation

### Recommended: Arch / CachyOS / BlackArch

The included installer sets up all system tools (nmap, hashcat, hydra, sqlmap, etc.), adds the BlackArch repository, and installs ptCenter itself using `uv`.

```bash
git clone https://github.com/j0yb0y-m/ptcenter.git
cd ptcenter
chmod +x install_arch.sh

# Full install (adds BlackArch repo)
./install_arch.sh

# Skip BlackArch (if already configured or not needed)
./install_arch.sh --no-blackarch

# Choose AUR helper
./install_arch.sh --aur-helper yay
```

### Manual install with uv (any distro)

[`uv`](https://github.com/astral-sh/uv) is the recommended Python package manager — fast, reproducible, and handles optional extras cleanly.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via `pip`
pip install --user uv --break-system-packages

# Clone the repo
git clone https://github.com/j0yb0y-m/ptcenter.git
cd ptcenter

# Core + all AI providers + OSINT + network extras
uv sync --extra full

# Or pick individual extras:
uv sync --extra ai       # AI providers only (Gemini, OpenAI, Claude)
uv sync --extra osint    # OSINT extras (holehe, maigret, sherlock)
uv sync --extra network  # Network extras (scapy)
```

### pip fallback

```bash
git clone https://github.com/j0yb0y-m/ptcenter.git
cd ptcenter
pip install -r requirements.txt
```

### System tool dependencies

ptCenter wraps external tools. Install the ones you need:

| Tool      | Arch/CachyOS                         | Debian/Kali/Parrot                                                         | Fedora                                 |
| --------- | ------------------------------------ | -------------------------------------------------------------------------- | -------------------------------------- |
| nmap      | `sudo pacman -S nmap`                | `sudo apt install nmap`                                                    | `sudo dnf install nmap`                |
| gobuster  | `sudo pacman -S gobuster`            | `sudo apt install gobuster`                                                | `sudo dnf install gobuster`            |
| nikto     | `sudo pacman -S nikto`               | `sudo apt install nikto`                                                   | `sudo dnf install nikto`               |
| hydra     | `sudo pacman -S hydra`               | `sudo apt install hydra`                                                   | `sudo dnf install hydra`               |
| sqlmap    | `sudo pacman -S sqlmap`              | `sudo apt install sqlmap`                                                  | `sudo dnf install sqlmap`              |
| hashcat   | `sudo pacman -S hashcat`             | `sudo apt install hashcat`                                                 | `sudo dnf install hashcat`             |
| john      | `sudo pacman -S john`                | `sudo apt install john`                                                    | `sudo dnf install john`                |
| ffuf      | `sudo pacman -S ffuf`                | `go install github.com/ffuf/ffuf/v2/cmd/ffuf@latest`                       | same                                   |
| dalfox    | `sudo pacman -S dalfox`              | `go install github.com/hahwul/dalfox/v2@latest`                            | same                                   |
| subfinder | `sudo pacman -S subfinder`           | `go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest` | same                                   |
| amass     | `sudo pacman -S amass`               | `sudo apt install amass`                                                   | `sudo dnf install amass`               |
| exiftool  | `sudo pacman -S perl-image-exiftool` | `sudo apt install exiftool`                                                | `sudo dnf install perl-Image-ExifTool` |
| whatweb   | `sudo pacman -S whatweb`             | `sudo apt install whatweb`                                                 | `sudo dnf install whatweb`             |
| msfvenom  | `sudo pacman -S metasploit`          | `sudo apt install metasploit-framework`                                    | manual                                 |
| sslscan   | `sudo pacman -S sslscan`             | `sudo apt install sslscan`                                                 | `sudo dnf install sslscan`             |
| netexec   | `sudo pacman -S netexec`             | `uv add netexec`                                                           | `uv add netexec`                       |

> 💡 ptCenter automatically detects which tools are installed at startup (tool health check) and shows the correct install command for your distro when a tool is missing.

---

## Configuration

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

```ini
# .env — NEVER commit this file

# Google Gemini — free tier, recommended
# Get key: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_key_here

# OpenAI (optional, paid)
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o

# Anthropic Claude (optional, paid)
ANTHROPIC_API_KEY=
CLAUDE_MODEL=claude-3-5-haiku-latest

# Ollama — free, fully local/offline
# Install: https://ollama.com/download
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b  # fast & capable on limited hardware

# Default AI provider: gemini | openai | claude | ollama
ACTIVE_AI_MODEL=gemini

# Output directory (default: /tmp/ptcenter_outputs)
OUTPUT_DIR=

# Subprocess timeout in seconds (default: 300)
COMMAND_TIMEOUT=300

# Optional
SHODAN_API_KEY=
TRUECALLER_INSTALLATION_ID=
```

> ⚠️ `.env` is in `.gitignore` — it will never be committed. Double check before any push.

---

## Usage

### Interactive mode (default)

```bash
# If installed via uv / pip as a package
ptcenter

# Or run as a Python module
python -m ptcenter

# Skip tool health check on startup
ptcenter --no-check

# Custom output directory
ptcenter --output ~/pentest/htb-box
```

You will be greeted with the banner and the main menu:

```
 ╔══════════════════════════════════════════════════╗
 ║          🔒 Penetration Testing Centre 🔒         ║
 ║   Developer: Mahdi (@j0yb0y-m)  Version: 2.0    ║
 ║   AI: ✓ Google Gemini (gemini-2.0-flash)         ║
 ╚══════════════════════════════════════════════════╝

  [1]  Scanner          [8]  Post-Exploitation
  [2]  OSINT            [9]  AI Security Chat
  [3]  Vulnerability   [10]  Sessions
  [4]  Exploitation    [11]  Generate Report
  [5]  Network Attacks [12]  Settings
  [6]  Password Attacks[13]  Exit
  [7]  Web App Testing
```

### Non-interactive / CLI mode

```bash
# Run an nmap scan non-interactively
ptcenter --module scanner --tool nmap --target 10.10.10.1

# Vulnerability info lookup
ptcenter --module vuln --target CVE-2024-1234

# Combine with custom output dir
ptcenter --module scanner --tool nmap --target 10.10.14.0/24 --output ~/results
```

---

## Modules

For full module documentation, see **[docs/MODULES.md](docs/MODULES.md)**.

### [1] Scanner

Wraps the most common scanning and enumeration tools with input validation and automatic AI analysis.

| Sub-tool              | Binary                                  | Description                                                                                    |
| --------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------- |
| nmap                  | `nmap`                                  | 10 scan profiles: fast, full, version, OS, aggressive, stealth, UDP, vuln scripts, NSE, custom |
| Subdomain Scan        | `subfinder`, `amass`, `sublist3r`       | Passive & active subdomain discovery with first-level resolution                               |
| Directory Brute Force | `gobuster`, `ffuf`, `dirb`, `dirsearch` | Multi-tool directory & file enumeration                                                        |
| Nikto                 | `nikto`                                 | Web server vulnerability scanner with AI analysis                                              |
| SSL/TLS Scan          | `sslscan`, `testssl.sh`                 | Protocol, cipher, and certificate auditing                                                     |
| DNS Enumeration       | `dig`, `host`, `nslookup`               | A, MX, NS, TXT, CNAME, SPF, DMARC lookup                                                       |
| SMB Enumeration       | `enum4linux`, `netexec`, `crackmapexec` | Shares, users, OS, and policy enumeration                                                      |
| **Agentic Recon**     | AI-driven                               | AI analyses previous results and automatically decides + runs the best next scan               |

### [2] OSINT

| Sub-tool               | Description                                                           |
| ---------------------- | --------------------------------------------------------------------- |
| Email / Username Intel | Holehe account check + AI analysis of exposed services                |
| Domain Intelligence    | WHOIS, ASN, DNS records, geolocation, SSL cert transparency           |
| Phone Lookup           | Carrier, country, validity via `phonenumbers`; TrueCaller integration |
| Social Media Search    | Multi-platform username check via Sherlock / Maigret                  |
| Metadata Extraction    | ExifTool wrapper — GPS, camera, author, software metadata             |
| WHOIS Lookup           | Full WHOIS with parsed registrant, dates, nameservers                 |
| Shodan Search          | Live Shodan API query (requires API key)                              |

### [3] Vulnerability

CVE/vulnerability lookup with AI-powered contextual analysis, CVSS scoring, and exploitation guidance.

### [4] Exploitation

| Sub-tool                 | Description                                                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| Reverse Shell Generator  | 13 language templates (Bash, Python, PHP, Ruby, Perl, Go, Java, Node.js, PowerShell, Netcat…) with LHOST/LPORT substitution |
| Bind Shell Generator     | Listener-side shells                                                                                                        |
| msfvenom Payload Builder | Interactive msfvenom wrapper — format, platform, arch, encoder, LHOST/LPORT                                                 |
| Web Shell Generator      | PHP, ASP, JSP, ASPX one-liner and full shells                                                                               |
| SQL Injection Payloads   | Categorized payloads: auth bypass, UNION, blind, time-based, error-based, stacked, OOB                                      |
| XSS Payloads             | Reflected, stored, DOM, event-based, filter-bypass, polyglot payloads                                                       |

### [5] Network Attacks

> ⚠️ All network attack functions require explicit `I CONFIRM` prompt before execution. Use only on authorized networks.

ARP spoofing, DNS spoofing, DHCP starvation, SYN flood, SSL stripping, MITM setup, passive network sniffing, MAC flooding.

### [6] Password Attacks

| Sub-tool            | Description                                                                       |
| ------------------- | --------------------------------------------------------------------------------- |
| Hash Identification | `hashid` / `hash-identifier` with AI fallback                                     |
| Hash Cracking       | Hashcat & John the Ripper with mode selection, wordlist/rules                     |
| Service Brute Force | Hydra wrapper — SSH, FTP, HTTP, RDP, SMB, MySQL, PostgreSQL, VNC, Telnet          |
| Password Spray      | Single-password spray against a user list                                         |
| Wordlist Manager    | Locate rockyou.txt, list wordlist directories, merge/deduplicate custom wordlists |
| AI Hash Analysis    | AI identifies hash type and recommends cracking approach                          |

### [7] Web App Testing

| Sub-tool               | Description                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| SQLMap Integration     | Full SQLMap wrapper with cookie support, depth, level, risk                                        |
| Parameter Fuzzing      | ffuf-based parameter and endpoint discovery                                                        |
| XSS Scanner            | Dalfox automated XSS scanner                                                                       |
| CORS Check             | Pure-Python CORS misconfiguration detector                                                         |
| JWT Analyzer           | Header/payload decode, algorithm check, `alg:none` test, AI analysis                               |
| Security Headers Audit | Checks for HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy |
| Tech Fingerprinting    | WhatWeb + AI identification of stack, versions, and associated CVEs                                |

### [8] Post-Exploitation

> Reference and suggestion module — commands are displayed to the tester, nothing is executed remotely.

PrivEsc suggester (Linux / Windows), persistence mechanisms reference, lateral movement reference, data exfiltration techniques, credential harvesting locations, LOLBins reference.

---

## AI Integration

ptCenter supports four AI providers simultaneously. You can switch between them at runtime from the Settings menu.

| Provider             | Model                                    | Cost                              | Internet | Best For                 |
| -------------------- | ---------------------------------------- | --------------------------------- | -------- | ------------------------ |
| **Google Gemini**    | `gemini-2.0-flash`                       | **Free** (15 req/min, 1M tok/day) | Yes      | Recommended default      |
| **OpenAI**           | `gpt-4o` (configurable)                  | Paid                              | Yes      | Highest quality analysis |
| **Anthropic Claude** | `claude-3-5-haiku-latest` (configurable) | Paid                              | Yes      | Fast & accurate          |
| **Ollama**           | Any local model                          | **Free**                          | No       | Air-gapped / offline use |

> For Ollama on low-resource hardware, `qwen2.5:3b` is recommended. For pentest-specific local models try `ollama pull supergoatscriptguy/mythos-sec:8b`.

### AI capabilities

- **Scan analysis** — Every scan module optionally sends output to the AI for a structured security analysis (executive summary, vulnerabilities found, risk ratings, next steps).
- **Chunked analysis** — Outputs larger than ~7,500 chars are split, each chunk is summarized individually, then a final synthesis is generated. No output is too large to analyze.
- **Agentic recon** — Given current scan results, the AI responds with a JSON action `{"tool": "...", "flags": "...", "target": "...", "reason": "..."}` which ptCenter executes automatically, creating a self-driving recon chain.
- **Interactive chat** — A persistent conversation with the AI acting as an expert penetration tester. History is capped at 20 turns to avoid token overflow.
- **Report generation** — AI writes a 3-paragraph executive summary and a prioritized remediation checklist for your session's findings.

See **[docs/AI_SETUP.md](docs/AI_SETUP.md)** for detailed setup instructions.

---

## Session Management

Sessions let you track an entire engagement or CTF box from start to finish:

```
Session: HTB-Lame
Target:  10.10.10.3
Scope:   10.10.10.0/24
Status:  5 findings · 3 scans completed
```

- **Create** a session with a name, target IP/domain, and optional scope CIDRs.
- **Scope enforcement** — any scan targeting an out-of-scope IP prompts a `CONFIRM` override.
- **Findings** — modules automatically add findings to the active session (severity: Critical / High / Medium / Low / Info).
- **Save/Load** — sessions are persisted to `~/.ptcenter_sessions/` as JSON files, survives reboots.
- **Report** — generate an AI-enriched HTML report from any saved session.

---

## Reporting

Generate a professional, self-contained HTML report from any session:

```
[11] Generate Report
```

The report includes:
- Cover page with target, tester name, and scope
- AI-generated executive summary (3 paragraphs)
- Scope & methodology section
- Findings table with severity badges
- Individual finding cards with details and evidence
- AI-generated prioritized remediation checklist
- Completed scans log

Reports are fully self-contained HTML (no CDN dependencies) — works offline and can be sent directly to a client.

---

## Documentation

| Document | Description |
|---|---|
| [README.md](README.md) | This file — overview, installation, quickstart |
| [docs/MODULES.md](docs/MODULES.md) | Full module reference with all sub-tools and options |
| [docs/AI_SETUP.md](docs/AI_SETUP.md) | AI provider setup, Ollama model selection, troubleshooting |
| [docs/INSTALL.md](docs/INSTALL.md) | Detailed installation guide for all supported distros |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | How to contribute, add a module, or add an AI provider |

---

## License

ptCenter is released under the **GNU General Public License v3.0**.

```
Copyright (C) 2026  Mahdi (@j0yb0y-m)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```

See [LICENSE](LICENSE) for the full license text.

---

Made with ❤️ by **Mahdi aka J0yB0y**

[GitHub](https://github.com/j0yb0y-m) · For educational purposes and authorized testing only.