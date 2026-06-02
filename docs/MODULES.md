# ptCenter v2 — Module Reference

Full reference for all 8 modules, every sub-tool, their options, required binaries, and example flows.

---

## Module 1 — Scanner

**Menu entry:** `[1] Scanner`  
**Source file:** `ptcenter/modules/scanner.py`

All scanning functions run through `core.runner.run_command()` or `stream_command()` with `shell=False`. User-supplied targets are validated before any subprocess is invoked.

---

### 1.1 Nmap Scan

**Binary required:** `nmap`

Presents 10 pre-built scan profiles plus a fully custom option. After the scan completes, the XML output is parsed and optionally sent to the active AI model for analysis.

**Profiles:**

| # | Flags | Description |
|---|---|---|
| 1 | `-F -T4` | Fast scan — top 100 ports |
| 2 | `-p- -T4` | Full port scan — all 65535 ports |
| 3 | `-sV -T4` | Service & version detection |
| 4 | `-O -T4` | OS detection (requires root) |
| 5 | `-A -T4` | Aggressive — OS + version + scripts + traceroute |
| 6 | `-sS -T4` | SYN stealth scan (requires root) |
| 7 | `-sU -T4` | UDP scan (requires root, slow) |
| 8 | `--script vuln -T4` | NSE vulnerability scripts |
| 9 | `-sC -sV -T4` | Default NSE scripts + version |
| 10 | custom | Free-form flags — sanitized before use |

**Input validation:**
- Target must pass `validate_target()` — accepts IPv4, IPv6, CIDR ranges, and FQDNs.
- Custom flags are sanitized with `sanitize_nmap_flags()` — dangerous flags (`--interactive`, `-iL`, `--script-args`) are rejected.
- If a session is active and the target is outside the declared scope CIDRs, a `CONFIRM` override is required.

**Output:** XML + text saved to `<output_dir>/nmap_<target>_<timestamp>.xml` and `.txt`. AI analysis is appended to the text file.

---

### 1.2 Subdomain Scan

**Binaries tried (in order):** `subfinder`, `amass`, `sublist3r`

Runs passive subdomain discovery against the target domain, then attempts DNS resolution for each discovered subdomain to confirm liveness.

**Input validation:** Domain must pass `validate_domain()` — RFC-compliant label check (1–63 chars per label, max 253 total, alphanumeric + hyphen, no leading/trailing hyphens).

**Output:** `<output_dir>/subdomain_<domain>_<timestamp>.txt`

---

### 1.3 Directory Brute Force

**Binaries tried (in order):** `gobuster`, `ffuf`, `dirb`, `dirsearch`

Enumerates directories and files on a web server. Prompts for wordlist path (auto-locates `rockyou.txt` or common wordlist directories if blank).

**Input validation:** URL must pass `validate_url()` — must be `http://` or `https://` with a non-empty host.

**Output:** `<output_dir>/dirbrute_<timestamp>.txt`

---

### 1.4 Nikto Scan

**Binary required:** `nikto`

Runs a full Nikto web vulnerability scan with optional HTTPS flag, then sends results to the AI for categorized analysis.

**Output:** `<output_dir>/nikto_<timestamp>.txt`

---

### 1.5 SSL/TLS Scan

**Binaries tried (in order):** `sslscan`, `testssl.sh`

Enumerates supported TLS protocols, cipher suites, certificate details, and known weaknesses (BEAST, POODLE, Heartbleed, etc.).

**Input validation:** Accepts domain or `host:port` format.

**Output:** `<output_dir>/ssl_<timestamp>.txt`

---

### 1.6 DNS Enumeration

**Binaries tried (in order):** `dig`, `host`, `nslookup`

Queries A, AAAA, MX, NS, TXT, CNAME records. Also checks for SPF (`v=spf1`) and DMARC (`_dmarc.` prefix) records.

**Input validation:** Domain must pass `validate_domain()`.

**Output:** `<output_dir>/dns_<domain>_<timestamp>.txt`

---

### 1.7 SMB Enumeration

**Binaries tried (in order):** `enum4linux`, `netexec`, `crackmapexec`

Attempts to enumerate SMB shares, users, OS information, and domain/workgroup policy from a Windows/Samba target.

**Input validation:** Target must pass `validate_target()`.

**Output:** `<output_dir>/smb_<timestamp>.txt`

---

### 1.8 Agentic Recon *(AI-powered)*

**Requires:** Active AI model

The AI-driven autonomous recon loop. ptCenter runs an initial nmap scan, feeds the results to the AI, and asks it to decide the next best action as structured JSON:

```json
{
  "tool": "nikto",
  "flags": "-host http://10.10.10.1 -port 80",
  "target": "10.10.10.1",
  "reason": "Port 80 open with Apache 2.4.49 — checking for web vulnerabilities"
}
```

ptCenter validates the response, executes the suggested command, feeds the new results back to the AI, and repeats. The loop runs up to 5 iterations by default.

**Supported next-action tools:** `nmap`, `nikto`, `gobuster`, `sqlmap`, `sslscan`, `enum4linux`

**Output:** Full transcript saved to `<output_dir>/agentic_recon_<timestamp>.txt`

---

## Module 2 — OSINT

**Menu entry:** `[2] OSINT`  
**Source file:** `ptcenter/modules/osint.py`

All OSINT operations use `shell=False`. Network calls are made via `requests` or CLI tools wrapped in `run_command()`.

---

### 2.1 Email / Username Intelligence

**Binaries tried:** `holehe` (CLI), then programmatic fallback via `holehe` Python library + `httpx`/`trio`

**For email addresses:**
- Validates format with a strict regex.
- Runs Holehe to check if the email is registered on 100+ online services.
- AI analysis of discovered service footprint.

**For usernames (no @):**
- Falls back to Sherlock / Maigret for cross-platform username search.

**Output:** `<output_dir>/osint_email_<timestamp>.txt`

---

### 2.2 Domain Intelligence

Aggregates multiple passive recon sources for a target domain or IP:

- WHOIS registration data (via `whois` binary or `python-whois` fallback)
- IP geolocation + ASN lookup (ip-api.com)
- DNS record enumeration (A, MX, NS, TXT, SOA)
- SSL certificate info (expiry, issuer, subject)
- HTTP response headers analysis
- AI synthesis of all collected data

**Input validation:** Must pass `validate_domain()` or `validate_ip()`.

**Output:** `<output_dir>/domain_intel_<timestamp>.txt`

---

### 2.3 Phone Lookup

**Library:** `phonenumbers` (always available — core dependency)  
**Optional:** TrueCaller API (`TRUECALLER_INSTALLATION_ID` in `.env`)

- Parses the phone number into E.164 format.
- Extracts country code, carrier, region, and validity.
- Formats number in international, national, and RFC3966 formats.
- TrueCaller lookup if `TRUECALLER_INSTALLATION_ID` is configured.

**Output:** `<output_dir>/phone_<timestamp>.txt`

---

### 2.4 Social Media Search

**Binaries tried (in order):** `sherlock`, `maigret`

Cross-platform username enumeration across hundreds of sites.

- Sherlock: fast, broad coverage, well-maintained.
- Maigret: deeper profile building, more metadata extraction.

**Output:** `<output_dir>/social_<username>_<timestamp>.txt`

---

### 2.5 Metadata Extraction

**Binary required:** `exiftool` (perl-image-exiftool)

Accepts a local file path and extracts all embedded metadata:
- GPS coordinates (latitude / longitude)
- Camera make, model, software
- Document author, creation date, modification date
- Image dimensions, color space, compression

**Output:** `<output_dir>/metadata_<timestamp>.txt`

---

### 2.6 WHOIS Lookup

**Binary:** `whois`

Full WHOIS query with structured parsing. Extracted fields:
- Registrant name, organization, country
- Registration and expiry dates
- Registrar and nameservers
- Abuse contact

**Input validation:** Must pass `validate_domain()` or `validate_ip()`.

**Output:** `<output_dir>/whois_<timestamp>.txt`

---

### 2.7 Shodan Search

**Requires:** `SHODAN_API_KEY` in `.env`

Queries the Shodan API for a host or search term. Returns:
- Open ports and services
- Detected vulnerabilities (CVEs)
- OS and banner information
- Location and ISP

**Output:** `<output_dir>/shodan_<timestamp>.txt`

---

## Module 3 — Vulnerability

**Menu entry:** `[3] Vulnerability`  
**Source file:** `ptcenter/modules/vuln.py`

A focused vulnerability research and lookup module.

- Accepts CVE IDs (e.g. `CVE-2024-1234`) or a plain vulnerability name / keyword.
- Queries public vulnerability databases.
- Passes context to the AI for:
  - CVSS score interpretation
  - Affected versions and exploitation status
  - Proof-of-concept availability
  - Suggested remediation

**Output:** `<output_dir>/vuln_<timestamp>.txt`

---

## Module 4 — Exploitation

**Menu entry:** `[4] Exploitation`  
**Source file:** `ptcenter/modules/exploit.py`

Payload and shell generation. All commands are displayed to the tester — nothing is sent to a remote target by this module.

---

### 4.1 Reverse Shell Generator

Generates ready-to-use reverse shell one-liners for 13 languages/environments.

**Input validation:** LHOST must pass `validate_ip()`. LPORT must pass `validate_port()` (1–65535).

| # | Language/Environment |
|---|---|
| 1 | Bash TCP |
| 2 | Bash UDP |
| 3 | Python 2 |
| 4 | Python 3 |
| 5 | Netcat (traditional `-e`) |
| 6 | Netcat (OpenBSD mkfifo) |
| 7 | PHP |
| 8 | Ruby |
| 9 | Perl |
| 10 | PowerShell |
| 11 | Node.js |
| 12 | Golang |
| 13 | Java |

After generating, also shows the corresponding netcat listener command:

```bash
nc -lvnp <LPORT>
```

**Output:** `<output_dir>/revshell_<timestamp>.txt`

---

### 4.2 Bind Shell Generator

Generates bind shell commands (listener on the target) for Bash, Python, Netcat, and Socat.

**Output:** `<output_dir>/bindshell_<timestamp>.txt`

---

### 4.3 msfvenom Payload Builder

**Binary required:** `msfvenom`

Interactive msfvenom wrapper with prompts for:
- Payload (e.g. `linux/x64/shell_reverse_tcp`)
- LHOST, LPORT
- Output format (`elf`, `exe`, `raw`, `python`, `bash`, etc.)
- Platform and architecture
- Encoder and iterations

Constructs and runs the command with `shell=False`. Generated payload is saved to `<output_dir>/`.

**Output:** `<output_dir>/payload_<timestamp>.<format>`

---

### 4.4 Web Shell Generator

Generates web shells for common server-side languages:

| Type | Languages |
|---|---|
| One-liner | PHP, ASP, JSP, ASPX |
| Full shell | PHP (with file browse, command output) |

**Output:** `<output_dir>/webshell_<timestamp>.<ext>`

---

### 4.5 SQL Injection Payloads

Generates and displays a categorized payload library:

- Authentication bypass (`' OR '1'='1`, `admin'--`, etc.)
- UNION-based extraction
- Blind boolean-based
- Time-based blind (`SLEEP`, `WAITFOR DELAY`)
- Error-based
- Stacked queries
- Out-of-band (DNS exfiltration)

**Output:** `<output_dir>/sqli_payloads_<timestamp>.txt`

---

### 4.6 XSS Payloads

Generates categorized XSS payload library:

- Basic reflected (`<script>alert(1)</script>`)
- Event handlers (`onerror`, `onload`, `onfocus`)
- Filter bypass (case variation, HTML entities, null bytes)
- DOM-based
- Stored (persistent)
- Polyglots (single payload that fires in multiple contexts)

**Output:** `<output_dir>/xss_payloads_<timestamp>.txt`

---

## Module 5 — Network Attacks

**Menu entry:** `[5] Network Attacks`  
**Source file:** `ptcenter/modules/network.py`

> ⚠️ **Authorization required.** Every function in this module presents a mandatory `I CONFIRM` prompt before execution. The confirmation is logged to the active session. Use only on networks you own or have written permission to test.

Most functions require root privileges and/or `scapy` (`uv sync --extra network`).

---

| Sub-tool | Binary/Library | Description |
|---|---|---|
| ARP Spoofing | `arpspoof` / `scapy` | Poisons ARP cache of target + gateway |
| DNS Spoofing | `dnsspoof` / `scapy` | Responds to DNS queries with attacker-controlled IPs |
| DHCP Starvation | `yersinia` / `scapy` | Exhausts DHCP pool with spoofed MACs |
| SYN Flood | `hping3` / `scapy` | TCP SYN flood DoS |
| SSL Stripping | `sslstrip` | Downgrades HTTPS to HTTP in-path |
| MITM Setup | `arpspoof` + `iptables` | Full ARP MITM with IP forwarding |
| Network Sniffing | `tcpdump` / `scapy` | Passive packet capture with filter |
| MAC Flooding | `macof` / `scapy` | CAM table overflow to force hub-mode on switches |

---

## Module 6 — Password Attacks

**Menu entry:** `[6] Password Attacks`  
**Source file:** `ptcenter/modules/password.py`

---

### 6.1 Hash Identification

**Binaries tried (in order):** `hashid`, `hash-identifier`, then AI fallback

Identifies the hash algorithm of a pasted hash. When neither tool is installed, the AI model is prompted with the hash and asked to identify the type and suggest cracking modes.

**Output:** Displayed in terminal + `<output_dir>/hash_id_<timestamp>.txt`

---

### 6.2 Hash Cracking

**Binaries tried:** `hashcat`, `john`

Prompts for:
- Hash value or file path
- Hash mode (for hashcat) / format (for john)
- Wordlist path (auto-locates `rockyou.txt`)
- Rule file (optional)

Constructs and runs the crack command, streaming output in real time.

**Output:** `<output_dir>/crack_<timestamp>.txt`

---

### 6.3 Service Brute Force

**Binary required:** `hydra`

Prompts for target, service, username/username-list, and password list. Constructs a hydra command for the selected service.

**Supported services:** SSH, FTP, HTTP-GET, HTTP-POST, HTTPS, RDP, SMB, MySQL, PostgreSQL, VNC, Telnet, SMTP, POP3, IMAP

**Input validation:** Target must pass `validate_target()`. Port must pass `validate_port()`.

**Output:** `<output_dir>/bruteforce_<timestamp>.txt`

---

### 6.4 Password Spray

Single-password spray against a list of usernames for a given service. Uses `hydra` with `-u` (loop usernames first) to avoid account lockout patterns.

Prompts for: service, target, port, username list file, and single password.

**Output:** `<output_dir>/spray_<timestamp>.txt`

---

### 6.5 Wordlist Manager

Utilities for managing wordlists:

- Auto-locate `rockyou.txt` across common install paths
- List common wordlist directories (`/usr/share/wordlists/`, `/usr/share/seclists/`, etc.)
- Merge multiple wordlists and deduplicate

---

### 6.6 AI Hash Analysis

Sends the hash directly to the active AI model for:
- Type identification with confidence levels
- Recommended hashcat mode (`-m` value)
- Suggested cracking strategies (wordlist → rules → brute)

---

## Module 7 — Web App Testing

**Menu entry:** `[7] Web App Testing`  
**Source file:** `ptcenter/modules/webapp.py`

All URL inputs are validated through `validate_url()` before use.

---

### 7.1 SQLMap Integration

**Binary required:** `sqlmap`

Full interactive SQLMap wrapper. Prompts for:
- Target URL
- Cookie/session header
- Crawl depth (`--crawl`)
- Level (1–5) and Risk (1–3)

Streams output in real time. AI analysis of discovered injection points after completion.

**Output:** `<output_dir>/sqlmap_<timestamp>.txt`

---

### 7.2 Parameter Fuzzing

**Binary required:** `ffuf`

Fuzz URL parameters for hidden endpoints or parameter pollution. Prompts for URL with `FUZZ` placeholder, wordlist, and filter options (status code, response size).

**Output:** `<output_dir>/fuzz_<timestamp>.txt`

---

### 7.3 XSS Scanner

**Binary required:** `dalfox`

Automated XSS scanning with dalfox. Supports:
- Basic GET parameter scanning
- DOM XSS detection
- Blind XSS endpoint configuration

**Output:** `<output_dir>/xss_scan_<timestamp>.txt`

---

### 7.4 CORS Check

**Library:** `requests` (pure-Python, no binary required)

Sends cross-origin requests with `Origin` headers set to attacker-controlled values and analyzes the response:

| Check | What it detects |
|---|---|
| `Access-Control-Allow-Origin: *` | Wildcard — any origin allowed |
| Reflected origin | Server echoes back the attacker's origin |
| `Access-Control-Allow-Credentials: true` + reflected | Critical — credentials exposed to attacker origin |
| `null` origin allowed | Sandbox bypass risk |

**Output:** `<output_dir>/cors_<timestamp>.txt`

---

### 7.5 JWT Analyzer

**Library:** `PyJWT` + `base64` (core dependencies)

Accepts a raw JWT token and performs:
- Header and payload decode (no signature required)
- Algorithm inspection (`alg` field — flags `none`, `HS256`, `RS256`)
- Expiry check (`exp` claim)
- `alg:none` attack attempt (forge unsigned token)
- AI analysis of claims and potential weaknesses

**Output:** `<output_dir>/jwt_<timestamp>.txt`

---

### 7.6 Security Headers Audit

**Library:** `requests` (pure-Python)

Fetches the response headers from the target URL and audits for the presence and correct configuration of:

| Header | Secure Value |
|---|---|
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` |
| `Content-Security-Policy` | Should be present and restrictive |
| `X-Frame-Options` | `DENY` or `SAMEORIGIN` |
| `X-Content-Type-Options` | `nosniff` |
| `Referrer-Policy` | `no-referrer` or `strict-origin` |
| `Permissions-Policy` | Should restrict sensitive APIs |

AI summarizes missing headers, their risk impact, and recommended values.

**Output:** `<output_dir>/headers_<timestamp>.txt`

---

### 7.7 Tech Stack Fingerprinting

**Binary required:** `whatweb`

Identifies the technology stack of a target web application:
- Web server and version
- CMS (WordPress, Drupal, Joomla, etc.)
- JavaScript frameworks
- Analytics platforms
- Server-side language

AI cross-references identified technologies with known CVEs and suggests targeted attack paths.

**Output:** `<output_dir>/tech_<timestamp>.txt`

---

## Module 8 — Post-Exploitation

**Menu entry:** `[8] Post-Exploitation`  
**Source file:** `ptcenter/modules/postexploit.py`

> This module is a **reference and suggester** — it shows commands and techniques to the tester. Nothing is executed on a remote target.

---

### 8.1 PrivEsc Suggester

Prompts for OS (Linux/Windows) and current user context, then displays:

**Linux:**
- LinPEAS / PEASS-ng download and execution commands
- Manual checklist: `sudo -l`, SUID binaries, cron jobs, writable files, kernel version

**Windows:**
- WinPEAS download command
- Manual checklist: `whoami /all`, `systeminfo`, service configs, registry autorun, unquoted service paths, AlwaysInstallElevated

AI generates additional, context-specific PrivEsc suggestions based on the provided user info.

---

### 8.2 Persistence Reference

Categorized persistence technique reference for Linux and Windows:

**Linux:** cron jobs, systemd services, `.bashrc`/`.profile`, SSH authorized keys, LD_PRELOAD, `/etc/init.d`

**Windows:** Registry run keys, scheduled tasks, startup folder, WMI event subscriptions, DLL hijacking

---

### 8.3 Lateral Movement Reference

Command references for:
- Pass-the-Hash (PtH) with `netexec` / `impacket`
- Pass-the-Ticket (PtT) with `rubeus` / `mimikatz`
- Remote execution: `psexec.py`, `wmiexec.py`, `smbexec.py`
- SSH forwarding and tunneling (`ssh -L`, `-R`, `-D`)
- RDP pivoting

AI generates context-specific lateral movement paths based on provided information (discovered hosts, credentials).

---

### 8.4 Data Exfiltration Helper

Reference techniques for exfiltrating data:

- HTTP/HTTPS: `curl`, `wget` POST to listener
- DNS exfiltration: base64-encoded data in DNS queries
- ICMP tunneling
- SMB: `smbclient` / `copy` to share
- Base64 encode/decode

---

### 8.5 Credential Harvesting

Common credential locations reference for Linux and Windows:

**Linux:** `/etc/shadow`, `.bash_history`, SSH keys, browser credential stores, application config files, environment variables, memory (`/proc/*/mem`)

**Windows:** SAM database, LSASS (mimikatz), credential manager, browser passwords, registry, PowerShell history

AI suggests additional harvesting paths based on provided system context.

---

### 8.6 LOLBins Reference

Living-off-the-land binaries reference:

**Windows LOLBins:** `certutil`, `bitsadmin`, `regsvr32`, `mshta`, `wscript`, `cscript`, `rundll32`, `powershell`

**Linux LOLBins (GTFOBins):** `python`, `perl`, `ruby`, `nc`, `ncat`, `curl`, `wget`, `find`, `tar`, `less`, `vim`

For each binary: download, execute, reverse shell, and file write capabilities. AI provides context-specific LOLBin chains.

---

## AI Security Chat

**Menu entry:** `[9] AI Security Chat`

A persistent, history-aware security assistant REPL. The AI is system-prompted as an expert penetration tester and CTF player.

```
🤖 AI Security Assistant  (type 'exit' or 'quit' to leave, 'clear' to reset)
System: Expert penetration tester & CTF player

You: I found port 6379 open, what should I try?
AI:  Port 6379 is Redis. First check if it requires auth:
     redis-cli -h <target> ping
     ...
```

**Commands inside chat:**
- `exit` / `quit` — return to main menu
- `clear` — reset conversation history

**History:** Capped at the last 20 turns (40 role/content pairs) to prevent token overflow. Older turns are trimmed from the front.
