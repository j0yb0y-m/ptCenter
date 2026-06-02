"""
ptcenter.modules.vuln
=====================
Vulnerability information: NVD CVE lookup + AI deep analysis.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from rich.console import Console
from rich.table import Table

console = Console()


def vulnerability_info(output_dir: Path, ai_manager: Any = None) -> None:
    vuln_id = console.input("[bold white][+] CVE ID or vulnerability name: [/]").strip()
    if not vuln_id:
        console.print("[red]✗[/] Input cannot be empty")
        return

    if vuln_id.upper().startswith("CVE-"):
        fetch_cve_info(vuln_id)

    if not ai_manager or not ai_manager.is_available():
        console.print("[yellow]⚠[/] No AI model available — configure an API key")
        return

    console.print(f"[bold blue]🤖[/] Analysing {vuln_id} with AI…")

    prompt = (
        f"Provide a comprehensive security analysis for: {vuln_id}\n\n"
        "1. Vulnerability Description and Technical Details\n"
        "2. Affected Systems / Software / Versions\n"
        "3. CVSS Score and Severity Rating\n"
        "4. Attack Vector and Complexity\n"
        "5. Potential Impact (CIA triad)\n"
        "6. Exploitation Status (known exploits, PoCs, active exploitation)\n"
        "7. Mitigation Strategies and Patches\n"
        "8. Detection Methods\n"
        "9. Related Vulnerabilities\n"
        "10. Security Recommendations\n\n"
        "Format for terminal display with clear sections."
    )

    response = ai_manager.generate(
        prompt,
        "You are a cybersecurity expert specialising in vulnerability analysis. "
        "Provide accurate, detailed, and actionable information.",
    )

    if response:
        ai_manager.display_analysis(response, f"Vulnerability Analysis: {vuln_id}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = output_dir / f"vuln_{vuln_id.replace('/', '_')}_{ts}.txt"
        with open(out, "w") as fh:
            fh.write(f"Vulnerability Analysis: {vuln_id}\n{datetime.now()}\n{'='*60}\n\n{response}")
        console.print(f"[bold green]✓[/] Saved: [cyan]{out}[/]")


def fetch_cve_info(cve_id: str) -> None:
    console.print(f"[bold blue]▶[/] Fetching {cve_id} from NVD…")
    try:
        url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            console.print(f"[yellow]⚠[/] NVD returned {resp.status_code}")
            return

        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            console.print("[yellow]⚠[/] CVE not found in NVD")
            return

        vuln = vulns[0]["cve"]
        table = Table(title=f"[bold green]NVD: {vuln.get('id','N/A')}[/]")
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        table.add_row("Published", vuln.get("published", "N/A")[:10])
        table.add_row("Modified", vuln.get("lastModified", "N/A")[:10])

        descs = vuln.get("descriptions", [])
        if descs:
            table.add_row("Description", descs[0]["value"][:300])

        metrics = vuln.get("metrics", {})
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics:
                cvss = metrics[key][0]["cvssData"]
                table.add_row("CVSS Score", str(cvss.get("baseScore", "N/A")))
                table.add_row("Severity", cvss.get("baseSeverity", "N/A"))
                table.add_row("Vector", cvss.get("vectorString", "N/A"))
                break

        console.print(table)

    except Exception as exc:
        console.print(f"[yellow]⚠[/] Could not fetch CVE data: {exc}")
