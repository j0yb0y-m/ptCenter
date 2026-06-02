"""
ptcenter.core.sysinfo
=====================
OS/distro detection and package-manager abstraction.

Supports:
  - Arch Linux, CachyOS, BlackArch, Manjaro, EndeavourOS, Garuda, Artix
  - Debian, Ubuntu, Kali, Parrot, Linux Mint
  - Fedora / RHEL family (basic)

Call ``install_hint("nmap")`` anywhere to get the right install command
for the currently running distro without hardcoding ``apt`` strings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

DistroFamily = Literal["arch", "debian", "fedora", "unknown"]

# ------------------------------------------------------
# Package name / install-command map
# Keys are the binary name (as passed to shutil.which).
# Each entry maps family → full install command string.
# ------------------------------------------------------
_PKG_MAP: dict[str, dict[str, str]] = {
    # ── Scanning ─────────────────────────────────────────────────────────────
    "nmap":         {"arch": "sudo pacman -S nmap",
                     "debian": "sudo apt install nmap",
                     "fedora": "sudo dnf install nmap"},
    "gobuster":     {"arch": "sudo pacman -S gobuster",          # blackarch / AUR
                     "debian": "sudo apt install gobuster",
                     "fedora": "sudo dnf install gobuster"},
    "nikto":        {"arch": "sudo pacman -S nikto",             # blackarch / AUR
                     "debian": "sudo apt install nikto",
                     "fedora": "sudo dnf install nikto"},
    "sqlmap":       {"arch": "sudo pacman -S sqlmap",            # blackarch / AUR
                     "debian": "sudo apt install sqlmap",
                     "fedora": "sudo dnf install sqlmap"},
    "ffuf":         {"arch": "sudo pacman -S ffuf",              # blackarch / AUR
                     "debian": "go install github.com/ffuf/ffuf/v2/cmd/ffuf@latest",
                     "fedora": "go install github.com/ffuf/ffuf/v2/cmd/ffuf@latest"},
    "dalfox":       {"arch": "sudo pacman -S dalfox",            # blackarch / AUR
                     "debian": "go install github.com/hahwul/dalfox/v2@latest",
                     "fedora": "go install github.com/hahwul/dalfox/v2@latest"},
    "amass":        {"arch": "sudo pacman -S amass",             # blackarch / AUR
                     "debian": "sudo apt install amass",
                     "fedora": "sudo dnf install amass"},
    "subfinder":    {"arch": "sudo pacman -S subfinder",         # blackarch / AUR
                     "debian": "go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest",
                     "fedora": "go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest"},
    "sublist3r":    {"arch": "sudo pacman -S sublist3r",         # blackarch / AUR
                     "debian": "uv add sublist3r",
                     "fedora": "uv add sublist3r"},
    "dirb":         {"arch": "sudo pacman -S dirb",              # blackarch / AUR
                     "debian": "sudo apt install dirb",
                     "fedora": "sudo dnf install dirb"},
    "dirsearch":    {"arch": "sudo pacman -S dirsearch",         # blackarch / AUR
                     "debian": "uv add dirsearch",
                     "fedora": "uv add dirsearch"},
    "sslscan":      {"arch": "sudo pacman -S sslscan",           # AUR
                     "debian": "sudo apt install sslscan",
                     "fedora": "sudo dnf install sslscan"},
    "testssl.sh":   {"arch": "sudo pacman -S testssl",           # blackarch / AUR
                     "debian": "sudo apt install testssl.sh",
                     "fedora": "sudo dnf install testssl"},
    "whatweb":      {"arch": "sudo pacman -S whatweb",           # blackarch / AUR
                     "debian": "sudo apt install whatweb",
                     "fedora": "sudo dnf install whatweb"},
    "netexec":      {"arch": "sudo pacman -S netexec",           # blackarch / AUR
                     "debian": "uv add netexec",
                     "fedora": "uv add netexec"},
    "crackmapexec": {"arch": "sudo pacman -S crackmapexec",      # blackarch / AUR
                     "debian": "uv add crackmapexec",
                     "fedora": "uv add crackmapexec"},
    "enum4linux":   {"arch": "sudo pacman -S enum4linux",        # blackarch / AUR
                     "debian": "sudo apt install enum4linux",
                     "fedora": "uv add enum4linux"},
    # ── Passwords ────────────────────────────────────────────────────────────
    "hydra":        {"arch": "sudo pacman -S thc-hydra",         # official / blackarch
                     "debian": "sudo apt install hydra",
                     "fedora": "sudo dnf install hydra"},
    "john":         {"arch": "sudo pacman -S john",
                     "debian": "sudo apt install john",
                     "fedora": "sudo dnf install john"},
    "hashcat":      {"arch": "sudo pacman -S hashcat",
                     "debian": "sudo apt install hashcat",
                     "fedora": "sudo dnf install hashcat"},
    "crunch":       {"arch": "sudo pacman -S crunch",            # blackarch / AUR
                     "debian": "sudo apt install crunch",
                     "fedora": "sudo dnf install crunch"},
    # ── Exploitation ─────────────────────────────────────────────────────────
    "msfvenom":     {"arch": "sudo pacman -S metasploit",        # blackarch / AUR
                     "debian": "sudo apt install metasploit-framework",
                     "fedora": "sudo dnf install metasploit"},
    # ── Network attacks ───────────────────────────────────────────────────────
    "arpspoof":     {"arch": "sudo pacman -S dsniff",            # AUR
                     "debian": "sudo apt install dsniff",
                     "fedora": "sudo dnf install dsniff"},
    "macof":        {"arch": "sudo pacman -S dsniff",            # AUR
                     "debian": "sudo apt install dsniff",
                     "fedora": "sudo dnf install dsniff"},
    "dnsspoof":     {"arch": "sudo pacman -S dsniff",            # AUR
                     "debian": "sudo apt install dsniff",
                     "fedora": "sudo dnf install dsniff"},
    "ettercap":     {"arch": "sudo pacman -S ettercap",
                     "debian": "sudo apt install ettercap-text-only",
                     "fedora": "sudo dnf install ettercap"},
    "hping3":       {"arch": "sudo pacman -S hping",             # AUR (binary stays hping3)
                     "debian": "sudo apt install hping3",
                     "fedora": "sudo dnf install hping3"},
    "sslstrip":     {"arch": "paru -S sslstrip",                 # AUR
                     "debian": "uv add sslstrip",
                     "fedora": "uv add sslstrip"},
    "mitmproxy":    {"arch": "sudo pacman -S mitmproxy",
                     "debian": "uv add mitmproxy",
                     "fedora": "sudo dnf install mitmproxy"},
    "tcpdump":      {"arch": "sudo pacman -S tcpdump",
                     "debian": "sudo apt install tcpdump",
                     "fedora": "sudo dnf install tcpdump"},
    "wireshark":    {"arch": "sudo pacman -S wireshark-qt",
                     "debian": "sudo apt install wireshark",
                     "fedora": "sudo dnf install wireshark"},
    "tshark":       {"arch": "sudo pacman -S wireshark-cli",
                     "debian": "sudo apt install tshark",
                     "fedora": "sudo dnf install wireshark-cli"},
    "yersinia":     {"arch": "paru -S yersinia",                 # AUR
                     "debian": "sudo apt install yersinia",
                     "fedora": "paru -S yersinia"},
    # ── OSINT ────────────────────────────────────────────────────────────────
    "exiftool":     {"arch": "sudo pacman -S perl-image-exiftool",
                     "debian": "sudo apt install libimage-exiftool-perl",
                     "fedora": "sudo dnf install perl-Image-ExifTool"},
    "shodan":       {"arch": "paru -S shodan",                   # AUR
                     "debian": "uv add shodan",
                     "fedora": "uv add shodan"},
    "sherlock":     {"arch": "sudo pacman -S sherlock",          # blackarch / AUR
                     "debian": "uv add sherlock-project",
                     "fedora": "uv add sherlock-project"},
    "maigret":      {"arch": "paru -S maigret",                  # AUR
                     "debian": "uv add maigret",
                     "fedora": "uv add maigret"},
    "holehe":       {"arch": "paru -S holehe",                   # AUR
                     "debian": "uv add holehe",
                     "fedora": "uv add holehe"},
    # ── System / DNS ────────────────────────────────────────────────────────
    "dig":          {"arch": "sudo pacman -S bind",
                     "debian": "sudo apt install dnsutils",
                     "fedora": "sudo dnf install bind-utils"},
    "whois":        {"arch": "sudo pacman -S whois",
                     "debian": "sudo apt install whois",
                     "fedora": "sudo dnf install whois"},
    "git":          {"arch": "sudo pacman -S git",
                     "debian": "sudo apt install git",
                     "fedora": "sudo dnf install git"},
}

# Wordlist search paths per family (checked in order; first existing path wins)
_WORDLIST_SEARCH: dict[str, list[str]] = {
    "arch": [
        "/usr/share/wordlists/rockyou.txt",           # blackarch wordlists pkg
        "/usr/share/seclists/Passwords/Leaked-Databases/rockyou.txt",
        str(Path.home() / "wordlists/rockyou.txt"),
        str(Path.home() / "SecLists/Passwords/Leaked-Databases/rockyou.txt"),
    ],
    "debian": [
        "/usr/share/wordlists/rockyou.txt",
        str(Path.home() / "wordlists/rockyou.txt"),
    ],
}
_WORDLIST_SEARCH["fedora"] = _WORDLIST_SEARCH["debian"]
_WORDLIST_SEARCH["unknown"] = _WORDLIST_SEARCH["debian"]

_WORDLIST_DIRS_SEARCH: dict[str, list[str]] = {
    "arch": [
        "/usr/share/wordlists",
        "/usr/share/seclists",
        str(Path.home() / "wordlists"),
        str(Path.home() / "SecLists"),
    ],
    "debian": [
        "/usr/share/wordlists",
        str(Path.home() / "wordlists"),
    ],
}
_WORDLIST_DIRS_SEARCH["fedora"] = _WORDLIST_DIRS_SEARCH["debian"]
_WORDLIST_DIRS_SEARCH["unknown"] = _WORDLIST_DIRS_SEARCH["debian"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def detect_family() -> DistroFamily:
    """Detect distro family by reading /etc/os-release (cached after first call)."""
    _ARCH_IDS = {"arch", "cachyos", "blackarch", "manjaro", "endeavouros",
                 "garuda", "artix", "parabola", "crystal", "archcraft"}
    _DEB_IDS  = {"debian", "ubuntu", "kali", "parrot", "linuxmint",
                 "raspbian", "pop", "elementary", "zorin"}
    _RPM_IDS  = {"fedora", "rhel", "centos", "rocky", "almalinux",
                 "opensuse", "sles"}
    try:
        text = Path("/etc/os-release").read_text().lower()
        for line in text.splitlines():
            if line.startswith("id=") or line.startswith("id_like="):
                val = line.split("=", 1)[1].strip().strip('"').replace("-", "")
                ids = set(val.split())
                if ids & _ARCH_IDS:
                    return "arch"
                if ids & _DEB_IDS:
                    return "debian"
                if ids & _RPM_IDS:
                    return "fedora"
        # Fallback: full-text scan
        if any(k in text for k in _ARCH_IDS):
            return "arch"
        if any(k in text for k in _DEB_IDS):
            return "debian"
        if any(k in text for k in _RPM_IDS):
            return "fedora"
    except Exception:
        pass
    return "unknown"


def is_arch() -> bool:
    return detect_family() == "arch"


def is_debian() -> bool:
    return detect_family() == "debian"


def distro_name() -> str:
    """Return a human-readable distro name, e.g. 'CachyOS', 'Kali Linux'."""
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return "Linux"


def install_hint(tool: str) -> str:
    """
    Return the correct install command string for *tool* on this distro.

    >>> install_hint("nmap")        # on CachyOS → "sudo pacman -S nmap"
    >>> install_hint("nmap")        # on Kali    → "sudo apt install nmap"
    """
    family = detect_family()
    mapping = _PKG_MAP.get(tool, {})
    if family in mapping:
        return mapping[family]
    # Fallback chain: unknown → debian hint, or bare tool name
    return mapping.get("debian", mapping.get("arch", f"install {tool}"))


def rockyou_path() -> str:
    """Return the path to rockyou.txt for this distro (first existing file)."""
    family = detect_family()
    for p in _WORDLIST_SEARCH.get(family, _WORDLIST_SEARCH["debian"]):
        if Path(p).exists():
            return p
    return "/usr/share/wordlists/rockyou.txt"  # conventional default


def wordlist_dirs() -> list[Path]:
    """Return existing wordlist search directories for this distro."""
    family = detect_family()
    paths = _WORDLIST_DIRS_SEARCH.get(family, _WORDLIST_DIRS_SEARCH["debian"])
    return [Path(p) for p in paths if Path(p).exists()]
