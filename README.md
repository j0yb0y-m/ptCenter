# ptCenter 🔒

> **AI-Powered Penetration Testing Framework**  
> Version 2.0 — Multi-Model Edition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white) ![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Kali%20%7C%20Parrot-lightgrey?logo=linux) ![License](https://img.shields.io/badge/License-GPL--3.0-blue) ![AI](https://img.shields.io/badge/AI-Gemini%20%7C%20GPT--4o%20%7C%20Claude%20%7C%20Ollama-purple)

**ptCenter** is a terminal-based penetration testing toolkit that combines classic security tools with multi-model AI analysis. Run network scans, perform OSINT investigations, generate exploit payloads, and get instant AI-powered breakdowns — all from a single, colorized CLI.

---

## ✨ Features

### 🖥️ Network Scanning

- **Nmap** — TCP SYN, UDP, aggressive, and stealth scans
- **SSL/TLS Analysis** — Cipher suite and certificate inspection via `sslscan`
- **DNS Enumeration** — Full record-type queries (A, AAAA, MX, NS, TXT, SOA, CNAME) plus zone-transfer attempts
- **Subdomain Enumeration** — Sublist3r, Amass, and Subfinder with automatic result merging
- **Directory Brute Force** — Gobuster, Dirb, and Dirsearch
- **Nikto Web Scanner** — Misconfigurations, outdated software, and known CVEs
- **SMB Enumeration** — Shares, users, and policies via `enum4linux`

### 🔍 OSINT Investigation

- **Email / Username Intelligence** — Holehe account checker, breached-password lookup
- **Domain / IP Intelligence** — WHOIS, geolocation, BGP ASN, reverse DNS
- **Phone Number Lookup** — TrueCaller integration
- **Social Media Search** — Username recon across platforms via Maigret / Sherlock
- **Metadata Extraction** — EXIF and document metadata via ExifTool
- **Shodan Search** — Exposed devices and services

### 🛡️ Vulnerability Analysis

- **CVE Lookup** — Live data from the NVD (NIST) REST API
- **AI Analysis** — Detailed breakdown: CVSS score, attack vector, mitigations, related CVEs

### 💉 Exploit Development

- **Reverse / Bind Shell Generator** — Bash, Python, Netcat, Perl, PHP, Ruby, PowerShell, and more
- **Msfvenom Payload Generator** — Windows, Linux, and macOS payloads
- **Web Shell Generator** — PHP, ASP, JSP
- **SQL Injection Payloads** — Common bypass, UNION, error-based, blind payloads
- **XSS Payloads** — Reflected, stored, and DOM-based vectors

### 🌐 Network Attacks

- **ARP Spoofing / Poisoning**
- **DNS Spoofing**
- **MITM Interception**
- **DHCP Starvation & SYN Flood** — via Scapy

### 🤖 Multi-Model AI Engine

Switch between any combination of providers at runtime:

|Provider|Model|Cost|
|---|---|---|
|Google Gemini|`gemini-2.0-flash`|**Free tier**|
|OpenAI|`gpt-4o` (configurable)|Paid / free trial credits|
|Anthropic Claude|`claude-3-5-haiku-latest` (configurable)|Paid / free trial credits|
|Ollama|`llama3` (configurable)|**100% free & offline**|

---

## 📋 Requirements

### System

- **OS:** Linux (Kali, Parrot, Ubuntu, Debian)
- **Python:** 3.8 or higher
- **Privileges:** Root / `sudo` recommended for network-level features

### Recommended External Tools

Install the tools for the modules you plan to use:

```bash
# Core tools
sudo apt update && sudo apt install -y \
  nmap sslscan nikto dirb gobuster dirsearch \
  enum4linux dig whois exiftool sslscan

# Subdomain enumeration
sudo apt install -y amass
pip install sublist3r --break-system-packages
go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest

# Network attack tools (Scapy is installed via pip)
sudo apt install -y arpspoof dsniff

# Shodan CLI (optional)
pip install shodan --break-system-packages
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/j0yb0y-m/ptCenter.git
cd ptCenter
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

> **Note:** The `--break-system-packages` flag is required on newer Debian/Ubuntu/Kali systems. Use a virtual environment if you prefer (see Virtual Environment Setup Section).

### 3. Configure environment variables

```bash
cp env.example .env
nano .env        # or: vim .env / code .env
```

Fill in at least one AI model key (see AI Model Configuration Section).

### 4. Run

```bash
python3 ptCenter.py
```

Some features require root:

```bash
sudo python3 ptCenter.py
```

---

## ⚙️ AI Model Configuration

ptCenter supports four AI backends simultaneously. Open your `.env` file and configure whichever providers you want:

```env
# ── Free ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY="your-api-key"          # https://aistudio.google.com/app/apikey

# ── Paid (free trial credits available) ──────────────────────────────────────
OPENAI_API_KEY="your-api-key"              # https://platform.openai.com/api-keys
OPENAI_MODEL=gpt-4o                # optional; default: gpt-4o

ANTHROPIC_API_KEY="your-api-key"       # https://console.anthropic.com/
CLAUDE_MODEL=claude-3-5-haiku-latest  # optional

# ── 100% Offline ─────────────────────────────────────────────────────────────
OLLAMA_HOST=http://localhost:11434  # optional; default shown
OLLAMA_MODEL=llama3                 # optional; default: llama3

# ── Select default model at startup ─────────────────────────────────────────
ACTIVE_AI_MODEL=gemini              # gemini | openai | claude | ollama
```

All configured models load at startup. Switch between them anytime from **Settings → Change Active AI Model** inside the app.

> ⚠️ Never commit your `.env` file to version control. It's already in `.gitignore`.

### Getting API Keys

| Provider         | Free Tier        | Link                                                          |
| ---------------- | ---------------- | ------------------------------------------------------------- |
| Google Gemini    | ✅ Yes            | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| OpenAI           | Trial credits    | [platform.openai.com](https://platform.openai.com/api-keys)   |
| Anthropic Claude | Trial credits    | [console.anthropic.com](https://console.anthropic.com/)       |
| Ollama           | ✅ Free & offline | [ollama.com/download](https://ollama.com/download)            |

---

## 🦙 Ollama Setup (Offline AI)

Ollama lets you run large language models entirely on your local machine — no internet required after the initial model download.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (one-time download)
ollama pull llama3        # ~4 GB, recommended
# or
ollama pull mistral       # lighter alternative
ollama pull codellama     # optimised for code

# Start the server (runs automatically on most systems)
ollama serve
```

Then set in `.env`:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
```

---

## 🧩 Optional OSINT Libraries

These are loaded lazily — ptCenter starts fine without them. Install only what you need:

```bash
# Email account checker
pip install holehe trio httpx --break-system-packages

# Username OSINT
pip install maigret --break-system-packages
pip install sherlock-project --break-system-packages

# Phone number lookup (requires a TrueCaller account)
pip install truecallerpy --break-system-packages
truecallerpy login          # authenticate once
```

---

## 🗂️ Virtual Environment Setup

If you prefer to keep dependencies isolated:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp env.example .env
python3 ptCenter.py
```

---

## 📁 Project Structure

```
ptCenter/
├── ptCenter.py          # Main application
├── requirements.txt     # Python dependencies
├── env.example          # Environment variable template
├── .gitignore           # Excludes .env, logs, outputs
└── README.md
```

Output files are saved to `/tmp/ptcenter_outputs/` by default. Override with:

```env
OUTPUT_DIR=/your/custom/path
```

---

## 🖼️ Main Menu

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            MAIN MENU                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  1 ► Network Scanning       - Port, service, and vulnerability scanning │
│  2 ► OSINT Investigation    - Open source intelligence gathering        │
│  3 ► Vulnerability Info     - CVE lookup and security analysis          │
│  4 ► Exploit Development    - Shells, payloads, and exploits            │
│  5 ► Network Attacks        - ARP, DNS, MITM, and DoS attacks           │
│  6 ► Settings               - Configure AI models and options           │
│  7 ► Exit                                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Application Settings

| Environment Variable | Default                 | Description                     |
| -------------------- | ----------------------- | ------------------------------- |
| `OUTPUT_DIR`         | `/tmp/ptcenter_outputs` | Directory for scan output files |
| `COMMAND_TIMEOUT`    | `300`                   | Max seconds per shell command   |
| `ACTIVE_AI_MODEL`    | first available         | Startup AI model selection      |

Settings are also persisted in `~/.ptcenter_config.json` and can be toggled live from the **Settings** menu.

---

## ⚠️ Legal Disclaimer

ptCenter is intended **strictly for authorized security testing and educational purposes**.

- Only use ptCenter against systems and networks **you own** or have **explicit written permission** to test.
- Unauthorized use against third-party systems is illegal and unethical.
- The developer assumes **no liability** for misuse or damage caused by this tool.
- Always follow your local laws, ethical guidelines, and the scope of any engagement you are authorized for.

**Hack responsibly. 🔒**

---

## 👤 Author

**Mahdi** — [@j0yb0y-m](https://github.com/j0yb0y-m)

---

## Contributing

*Contributions, issues, and feature requests are welcome.*

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

You are free to use, modify, and distribute this software under the terms of the GPL-3.0. Any derivative works must also be distributed under the same license. See the [LICENSE](LICENSE) file for full details.
