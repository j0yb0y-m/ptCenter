# ptCenter v2 — Installation Guide

Detailed installation instructions for all supported Linux distributions.

---

## Table of Contents

- [Requirements](#requirements)
- [Arch Linux / CachyOS / BlackArch (Recommended)](#arch-linux--cachyos--blackarch-recommended)
- [Kali Linux / Parrot OS / Debian / Ubuntu](#kali-linux--parrot-os--debian--ubuntu)
- [Fedora / RHEL-based](#fedora--rhel-based)
- [Installing uv](#installing-uv)
- [Python Extras Reference](#python-extras-reference)
- [Verifying the Install](#verifying-the-install)
- [Updating ptCenter](#updating-ptcenter)
- [Uninstalling](#uninstalling)

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 or 3.12 recommended |
| OS | Linux | Tested on Arch, CachyOS, Kali, Parrot, Ubuntu |
| Disk | ~200MB | For ptCenter + core Python deps |
| RAM | 512MB | More needed if running Ollama locally |
| Network | Optional | Required only for cloud AI providers and OSINT modules |
| Root/sudo | Optional | Required for SYN scans, raw sockets, ARP operations |

ptCenter itself does not require root. Individual tools it wraps (nmap `-sS`, scapy, arpspoof) may require root.

---

## Arch Linux / CachyOS / BlackArch (Recommended)

The included `install_arch.sh` script handles everything: system tools, BlackArch repo, Python environment setup, and ptCenter itself.

### Option A: Automated installer

```bash
git clone https://github.com/j0yb0y-m/ptcenter.git
cd ptcenter
chmod +x install_arch.sh
```

**Full install** (adds BlackArch repo for tools like sqlmap, nikto, gobuster, dalfox, netexec, etc.):
```bash
./install_arch.sh
```

**Skip BlackArch** (if you already have it or only want official repo tools):
```bash
./install_arch.sh --no-blackarch
```

**Use `yay` instead of `paru`**:
```bash
./install_arch.sh --aur-helper yay
```

The script will:
1. Run `sudo pacman -Syu`
2. Add the BlackArch repository (if not skipped)
3. Install core tools from official repos: `nmap`, `hashcat`, `john`, `tcpdump`, `wireshark-qt`, `wireshark-cli`, `ettercap`, `mitmproxy`, `perl-image-exiftool`, `git`, `curl`, `wget`, `bind`, `whois`, `python`, `python-pip`, `go`
4. Install AUR tools via your chosen helper
5. Install `uv` and sync the Python package with `--extra full`

### Option B: Manual install on Arch/CachyOS

```bash
# 1 — Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Reload shell or:
source ~/.bashrc  # or ~/.zshrc

# 2 — Clone and install ptcenter
git clone https://github.com/j0yb0y-m/ptcenter.git
cd ptcenter
uv sync --extra full

# 3 — Configure
cp .env.example .env
# Edit .env with your API keys

# 4 — Install system tools (example subset)
sudo pacman -S --needed nmap gobuster nikto hydra sqlmap john hashcat ffuf amass exiftool whatweb sslscan

# BlackArch tools (if repo is added)
sudo pacman -S --needed dalfox netexec subfinder enum4linux sublist3r
```

### Running on Arch/CachyOS

```bash
# As a module (within the cloned directory)
uv run python -m ptcenter

# Or if installed as a package (after uv sync)
uv run ptcenter
```

---

## Kali Linux / Parrot OS / Debian / Ubuntu

Kali and Parrot come with most security tools pre-installed. You mainly need to set up the Python package.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### Clone and sync

```bash
git clone https://github.com/j0yb0y-m/ptcenter.git
cd ptcenter

# Core + AI + OSINT extras (recommended)
uv sync --extra ai --extra osint

# Everything (includes network/scapy)
uv sync --extra full
```

### Install missing system tools

Most are already on Kali/Parrot. For anything missing:

```bash
sudo apt update
sudo apt install -y \
    nmap gobuster nikto hydra sqlmap john hashcat \
    ffuf amass exiftool whatweb sslscan enum4linux \
    tcpdump wireshark-common mitmproxy whois curl wget

# Go-based tools
go install github.com/hahwul/dalfox/v2@latest
go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
```

### pip fallback (if uv is not available)

```bash
# Inside the cloned directory, create a venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running on Kali/Parrot/Debian

```bash
# With uv
uv run python -m ptcenter

# With venv
source .venv/bin/activate
python -m ptcenter
```

---

## Fedora / RHEL-based

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone
git clone https://github.com/j0yb0y-m/ptcenter.git
cd ptcenter
uv sync --extra ai --extra osint

# Core tools via dnf
sudo dnf install -y \
    nmap hydra sqlmap john hashcat nikto \
    gobuster amass sslscan whatweb whois curl wget \
    perl-Image-ExifTool tcpdump

# Go-based tools
go install github.com/ffuf/ffuf/v2/cmd/ffuf@latest
go install github.com/hahwul/dalfox/v2@latest
go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
```

---

## Installing uv

[uv](https://github.com/astral-sh/uv) is a fast Python package manager written in Rust, recommended over pip for ptCenter because it handles optional extras correctly.

```bash
# Official installer (all distros)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload your shell
source ~/.bashrc   # or ~/.zshrc if using zsh

# Verify
uv --version
```

### Alternative: pip install uv

```bash
pip install uv
```

---

## Python Extras Reference

ptCenter uses optional dependency groups to avoid installing heavy libraries you don't need.

| Extra | What it adds | Install command |
|---|---|---|
| (none) | Core only: `python-dotenv`, `requests`, `rich`, `jinja2`, `PyJWT`, `phonenumbers` | `uv sync` |
| `ai` | `google-genai`, `openai`, `anthropic` | `uv sync --extra ai` |
| `osint` | `holehe`, `trio`, `httpx`, `maigret`, `sherlock-project`, `truecallerpy` | `uv sync --extra osint` |
| `network` | `scapy` | `uv sync --extra network` |
| `full` | Everything above | `uv sync --extra full` |

**Examples:**

```bash
# Minimal install — core modules only, no AI, no OSINT extras
uv sync

# AI + core only (no OSINT Python packages)
uv sync --extra ai

# Everything
uv sync --extra full

# Add an extra later without re-syncing everything
uv add --extra osint holehe
```

---

## Verifying the Install

After installation, run a quick sanity check:

```bash
# Check Python and uv are working
uv run python --version

# Launch ptcenter and check the banner
uv run python -m ptcenter --no-check

# Or if installed as a package command
ptcenter --version
# → ptcenter 2.0
```

### Checking tool health

ptCenter performs a tool health check at startup by default. You'll see a table like:

```
Tool Health Check
╔═════════════════╦══════════╦══════════════════════════════════╗
║ Tool            ║ Status   ║ Install                          ║
╠═════════════════╬══════════╬══════════════════════════════════╣
║ nmap            ║ ✓ Found  ║                                  ║
║ gobuster        ║ ✓ Found  ║                                  ║
║ nikto           ║ ✗ Missing║ sudo pacman -S nikto             ║
║ hydra           ║ ✓ Found  ║                                  ║
╚═════════════════╩══════════╩══════════════════════════════════╝
```

Missing tools show the correct install command for your distro. Install them as needed — ptCenter works fine with any subset of tools installed.

---

## Updating ptCenter

```bash
cd ptcenter
git pull
uv sync --extra full   # re-sync dependencies after any pyproject.toml changes
```

---

## Uninstalling

ptCenter creates a few persistent paths outside the repo directory:

| Path | Contents |
|---|---|
| `~/.ptcenter_config.json` | Your settings (output dir, AI toggle, tester name) |
| `~/.ptcenter_sessions/` | Saved engagement sessions |
| `/tmp/ptcenter_outputs/` (or your `OUTPUT_DIR`) | Scan outputs and logs |

To remove everything:

```bash
# Remove the repo
rm -rf /path/to/ptcenter

# Remove user data (optional)
rm -f ~/.ptcenter_config.json
rm -rf ~/.ptcenter_sessions/

# Remove output directory (if using default)
rm -rf /tmp/ptcenter_outputs/
```
