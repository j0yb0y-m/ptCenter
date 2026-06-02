#!/usr/bin/env bash
# ptCenter v2 — Arch / CachyOS / BlackArch Setup Script
# Usage:
#   chmod +x install_arch.sh
#   ./install_arch.sh              # full install (BlackArch tools)
#   ./install_arch.sh --no-blackarch  # skip BlackArch repo setup
#   ./install_arch.sh --aur-helper paru  # choose AUR helper (paru/yay, default: paru)

set -euo pipefail

# Colour helpers 
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[*]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
error()   { echo -e "${RED}[✗]${RESET} $*"; exit 1; }

# Argument parsing
USE_BLACKARCH=true
AUR_HELPER="paru"

for arg in "$@"; do
    case "$arg" in
        --no-blackarch)   USE_BLACKARCH=false ;;
        --aur-helper)     shift; AUR_HELPER="$1" ;;
        --aur-helper=*)   AUR_HELPER="${arg#*=}" ;;
        -h|--help)
            echo "Usage: $0 [--no-blackarch] [--aur-helper paru|yay]"
            exit 0 ;;
    esac
done

# Distro check
if ! grep -qiE 'arch|cachyos|blackarch|manjaro|endeavour|garuda|artix' /etc/os-release 2>/dev/null; then
    warn "This script is designed for Arch-based distros. Proceeding anyway…"
fi

echo -e "\n${BOLD}${CYAN}ptCenter v2 — Arch/CachyOS/BlackArch Installer${RESET}\n"

# System update 
info "Updating system (pacman -Syu)…"
sudo pacman -Syu --noconfirm

# BlackArch repository 
if $USE_BLACKARCH; then
    if grep -q '\[blackarch\]' /etc/pacman.conf 2>/dev/null; then
        success "BlackArch repo already configured"
    else
        info "Adding BlackArch repository…"
        curl -O https://blackarch.org/strap.sh
        chmod +x strap.sh
        sudo bash strap.sh
        rm -f strap.sh
        sudo pacman -Sy --noconfirm
        success "BlackArch repo added"
    fi
fi

# Core system packages (official Arch repos) 
info "Installing core packages from official repos…"
sudo pacman -S --needed --noconfirm \
    nmap \
    hashcat \
    john \
    tcpdump \
    wireshark-qt \
    wireshark-cli \
    ettercap \
    mitmproxy \
    perl-image-exiftool \
    git \
    curl \
    wget \
    bind \
    whois \
    python \
    python-pip \
    go

success "Core packages installed"

# BlackArch security tools 
if $USE_BLACKARCH; then
    info "Installing security tools from BlackArch…"
    sudo pacman -S --needed --noconfirm \
        gobuster \
        nikto \
        sqlmap \
        thc-hydra \
        ffuf \
        dalfox \
        amass \
        subfinder \
        sublist3r \
        dirb \
        dirsearch \
        sslscan \
        testssl \
        whatweb \
        netexec \
        crackmapexec \
        enum4linux \
        crunch \
        sherlock \
        metasploit \
        hping \
        dsniff \
        wordlists 2>/dev/null || warn "Some BlackArch packages not found — continuing"
    success "BlackArch security tools installed"
else
    warn "Skipping BlackArch tools (--no-blackarch was passed)"
fi

# AUR helper setup 
if ! command -v "$AUR_HELPER" &>/dev/null; then
    info "Installing AUR helper: $AUR_HELPER…"
    TMP_DIR=$(mktemp -d)
    git clone "https://aur.archlinux.org/${AUR_HELPER}.git" "$TMP_DIR/$AUR_HELPER"
    (cd "$TMP_DIR/$AUR_HELPER" && makepkg -si --noconfirm)
    rm -rf "$TMP_DIR"
    success "$AUR_HELPER installed"
else
    success "$AUR_HELPER already installed"
fi

# AUR-only packages
info "Installing AUR packages via $AUR_HELPER…"
"$AUR_HELPER" -S --needed --noconfirm \
    holehe \
    maigret \
    truecallerpy \
    sslstrip \
    yersinia 2>/dev/null || warn "Some AUR packages failed — continuing"
success "AUR packages installed"

# Wordlists setup
info "Setting up wordlists…"
WORDLISTS_DIR="/usr/share/wordlists"
ROCKYOU="$WORDLISTS_DIR/rockyou.txt"
ROCKYOU_GZ="$WORDLISTS_DIR/rockyou.txt.gz"

if [ -f "$ROCKYOU" ]; then
    success "rockyou.txt already present at $ROCKYOU"
elif [ -f "$ROCKYOU_GZ" ]; then
    info "Decompressing rockyou.txt.gz…"
    sudo gunzip "$ROCKYOU_GZ"
    success "rockyou.txt decompressed"
else
    warn "rockyou.txt not found. Install the 'wordlists' BlackArch package:"
    warn "  sudo pacman -S wordlists"
fi

# uv installation
if ! command -v uv &>/dev/null; then
    info "Installing uv…"
    CURL_INSTALLATION = $(curl -LsSf connect-timeout 10 --max-time 40 https://astral.sh/uv/install.sh | sh)
    exec $CURL_INSTALLATION
    # Add to PATH for the rest of this script
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    success "uv installed"
    if ! $CURL_INSTALLATION; then
      warn "uv installation via curl failed — trying pip fallback…"
      info "Installing uv via pip…"
      python -m pip install --user uv --break-system-packages
      export PATH="$HOME/.local/bin:$PATH"
      success "uv installed via pip"
    else 
      success "uv installed via curl"
    fi
else 
    success "uv already installed"
   fi

# ptCenter Python env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
info "Setting up ptCenter Python environment in $SCRIPT_DIR…"
cd "$SCRIPT_DIR"

uv sync --extra ai
success "Core + AI dependencies installed"

# .env setup 
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn ".env created from template — add your GEMINI_API_KEY to .env"
    warn "  Free key: https://aistudio.google.com/app/apikey"
else
    success ".env already exists"
fi

# Summary
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  ptCenter v2 installation complete!${RESET}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${CYAN}Run:${RESET}       uv run python -m ptcenter"
echo -e "  ${CYAN}No-check:${RESET}  uv run python -m ptcenter --no-check"
echo -e "  ${CYAN}Global:${RESET}    uv tool install .  # adds 'ptcenter' to PATH"
echo ""
echo -e "  ${YELLOW}Edit .env and add your GEMINI_API_KEY for AI features${RESET}"
echo ""

# Quick tool health summary
echo -e "${BOLD}Tool availability:${RESET}"
TOOLS=(nmap gobuster nikto hydra sqlmap john hashcat ffuf dalfox amass sqlmap whatweb)
for t in "${TOOLS[@]}"; do
    if command -v "$t" &>/dev/null; then
        echo -e "  ${GREEN}✓${RESET} $t"
    else
        echo -e "  ${RED}✗${RESET} $t"
    fi
done
echo ""
