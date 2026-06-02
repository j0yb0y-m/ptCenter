"""
ptcenter.core.reporter
======================
Generate a professional, self-contained HTML engagement report from a session.
Uses only stdlib + Jinja2.  No external CDN dependencies — works fully offline.
"""

from __future__ import annotations

import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, BaseLoader

from rich.console import Console

console = Console()

# Jinja2 template (inline — no template files required)

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ session.name }} — Penetration Test Report</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;line-height:1.6}
a{color:#58a6ff}
h1,h2,h3{color:#f0f6fc;font-weight:600}
h1{font-size:2rem;margin-bottom:.5rem}
h2{font-size:1.4rem;border-bottom:1px solid #30363d;padding-bottom:.4rem;margin:2rem 0 1rem}
h3{font-size:1.1rem;margin:1.2rem 0 .4rem}
.container{max-width:1100px;margin:0 auto;padding:2rem}
.cover{text-align:center;padding:4rem 2rem;border-bottom:2px solid #00b4d8}
.cover .subtitle{color:#8b949e;font-size:1rem;margin-top:.5rem}
.badge{display:inline-block;padding:.2rem .7rem;border-radius:4px;font-size:.75rem;font-weight:700;text-transform:uppercase}
.crit{background:#b62324;color:#fff}.high{background:#d1242f;color:#fff}
.med{background:#9e6a03;color:#fff}.low{background:#347d39;color:#fff}.info{background:#388bfd;color:#fff}
table{width:100%;border-collapse:collapse;margin:1rem 0}
th{background:#161b22;color:#8b949e;text-align:left;padding:.6rem .8rem;border:1px solid #30363d;font-size:.8rem;text-transform:uppercase}
td{padding:.6rem .8rem;border:1px solid #30363d;vertical-align:top}
tr:hover td{background:#161b22}
.finding-card{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:1.2rem;margin:1rem 0}
.finding-card h3{color:#58a6ff}
pre{background:#010409;border:1px solid #30363d;border-radius:4px;padding:1rem;overflow-x:auto;font-size:.82rem;white-space:pre-wrap;word-break:break-all;margin:.8rem 0}
.meta{color:#8b949e;font-size:.82rem}
.section-intro{color:#8b949e;margin-bottom:1rem}
.toc a{color:#58a6ff;text-decoration:none;display:block;padding:.2rem 0}
.toc a:hover{color:#79c0ff}
footer{text-align:center;padding:2rem;color:#8b949e;font-size:.75rem;border-top:1px solid #30363d;margin-top:3rem}
.pill{display:inline-block;background:#21262d;border:1px solid #30363d;border-radius:12px;padding:.1rem .6rem;font-size:.78rem;margin:.1rem}
.exec-summary{background:#161b22;border-left:4px solid #00b4d8;padding:1rem 1.2rem;border-radius:0 6px 6px 0;margin:1rem 0}
</style>
</head>
<body>
<div class="container">

<!-- COVER PAGE -->
<div class="cover">
  <div style="color:#00b4d8;font-size:.9rem;letter-spacing:.15em;text-transform:uppercase">Penetration Test Report</div>
  <h1>{{ session.name }}</h1>
  <div class="subtitle">Target: <strong>{{ session.target }}</strong></div>
  <div class="subtitle" style="margin-top:.3rem">Tester: {{ tester_name }}</div>
  <div class="subtitle">Report generated: {{ generated_at }}</div>
  <div style="margin-top:1.5rem">
    <span class="badge" style="background:#21262d;color:#c9d1d9;border:1px solid #30363d">
      Scope: {{ session.scope | join(', ') or 'All' }}
    </span>
  </div>
</div>

<!-- TOC -->
<h2 id="toc">Table of Contents</h2>
<div class="toc">
  <a href="#executive-summary">1. Executive Summary</a>
  <a href="#scope">2. Scope &amp; Methodology</a>
  <a href="#findings-table">3. Findings Overview</a>
  <a href="#detailed-findings">4. Detailed Findings</a>
  <a href="#recommendations">5. Recommendations</a>
  <a href="#appendix">6. Appendix</a>
</div>

<!-- EXECUTIVE SUMMARY -->
<h2 id="executive-summary">1. Executive Summary</h2>
<div class="exec-summary">
  <pre style="background:transparent;border:none;padding:0;font-family:inherit;font-size:.92rem">{{ executive_summary }}</pre>
</div>
<p class="meta">
  Total findings: <strong>{{ findings | length }}</strong> —
  {% for sev, count in severity_counts.items() %}
    <span class="pill">{{ sev }}: {{ count }}</span>
  {% endfor %}
</p>

<!-- SCOPE -->
<h2 id="scope">2. Scope &amp; Methodology</h2>
<p class="section-intro">This assessment was conducted against the following targets and networks:</p>
<ul style="margin-left:1.5rem">
  {% for item in (session.scope or [session.target]) %}
  <li>{{ item }}</li>
  {% endfor %}
</ul>
<h3>Scans Performed</h3>
<p>{% for scan in (session.scans_completed or []) %}<span class="pill">{{ scan }}</span>{% else %}<em>None recorded</em>{% endfor %}</p>
<h3>Engagement Dates</h3>
<p class="meta">Started: {{ session.created_at[:10] }}&emsp;Last updated: {{ session.updated_at[:10] }}</p>

<!-- FINDINGS TABLE -->
<h2 id="findings-table">3. Findings Overview</h2>
<table>
  <thead>
    <tr><th>#</th><th>Title</th><th>Severity</th><th>Module</th><th>Tool</th><th>Timestamp</th></tr>
  </thead>
  <tbody>
    {% for f in findings %}
    <tr>
      <td>{{ loop.index }}</td>
      <td><a href="#finding-{{ loop.index }}">{{ f.title }}</a></td>
      <td><span class="badge {{ severity_class_map[f.severity] }}">{{ f.severity }}</span></td>
      <td>{{ f.module }}</td>
      <td>{{ f.tool }}</td>
      <td class="meta">{{ f.timestamp[:19] }}</td>
    </tr>
    {% else %}
    <tr><td colspan="6" style="text-align:center;color:#8b949e">No findings recorded</td></tr>
    {% endfor %}
  </tbody>
</table>

<!-- DETAILED FINDINGS -->
<h2 id="detailed-findings">4. Detailed Findings</h2>
{% for f in findings %}
<div class="finding-card" id="finding-{{ loop.index }}">
  <h3>#{{ loop.index }} — {{ f.title }}</h3>
  <p class="meta">
    Severity: <span class="badge {{ severity_class_map[f.severity] }}">{{ f.severity }}</span>
    &emsp;Module: <span class="pill">{{ f.module }}</span>
    &emsp;Tool: <span class="pill">{{ f.tool }}</span>
    &emsp;Time: {{ f.timestamp[:19] }}
  </p>
  {% if f.output_file %}
  <p class="meta" style="margin-top:.5rem">Output file: <code>{{ f.output_file }}</code></p>
  {% endif %}
  <h3 style="margin-top:1rem">Analysis</h3>
  <pre>{{ f.details | e }}</pre>
</div>
{% else %}
<p class="section-intro">No detailed findings to display.</p>
{% endfor %}

<!-- RECOMMENDATIONS -->
<h2 id="recommendations">5. Recommendations</h2>
<div class="exec-summary">
  <pre style="background:transparent;border:none;padding:0;font-family:inherit;font-size:.92rem">{{ recommendations }}</pre>
</div>

<!-- APPENDIX -->
<h2 id="appendix">6. Appendix — Raw Output Files</h2>
{% if output_files %}
<table>
  <thead><tr><th>File</th><th>Path</th></tr></thead>
  <tbody>
    {% for fp in output_files %}
    <tr><td>{{ fp | replace(output_dir, '') }}</td><td class="meta"><code>{{ fp }}</code></td></tr>
    {% endfor %}
  </tbody>
</table>
{% else %}
<p class="section-intro">No output files recorded for this session.</p>
{% endif %}

<footer>
  Generated by ptCenter v2.0 &mdash; {{ generated_at }} &mdash; Authorized use only.
</footer>
</div>
</body>
</html>"""


# Public API

def generate_report(
    session: dict,
    output_dir: Path,
    tester_name: str = "Unknown",
    executive_summary: str = "",
    recommendations: str = "",
    open_browser: bool = True,
) -> Path:
    """
    Render the HTML report and write it to *output_dir*.
    Returns the path to the generated file.
    """
    env = Environment(loader=BaseLoader())
    tmpl = env.from_string(_TEMPLATE)

    # Collect output files from findings
    output_files: list[str] = []
    for f in session.get("findings", []):
        fp = f.get("output_file", "")
        if fp and fp not in output_files:
            output_files.append(fp)

    # Severity counts
    severity_order = ["Critical", "High", "Medium", "Low", "Info"]
    severity_counts: dict[str, int] = {s: 0 for s in severity_order}
    for f in session.get("findings", []):
        sev = f.get("severity", "Info")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    severity_counts = {k: v for k, v in severity_counts.items() if v > 0}

    severity_class_map = {
        "Critical": "crit",
        "High": "high",
        "Medium": "med",
        "Low": "low",
        "Info": "info",
    }

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = tmpl.render(
        session=session,
        findings=session.get("findings", []),
        tester_name=tester_name,
        generated_at=generated_at,
        executive_summary=executive_summary or "No executive summary generated.",
        recommendations=recommendations or "No recommendations generated.",
        output_files=output_files,
        output_dir=str(output_dir),
        severity_counts=severity_counts,
        severity_class_map=severity_class_map,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = session.get("name", "report").replace(" ", "_").replace("/", "_")
    report_path = output_dir / f"report_{safe_name}_{ts}.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    console.print(f"[bold green]✓[/] Report saved: [cyan]{report_path}[/]")

    if open_browser:
        try:
            webbrowser.open(f"file://{report_path.resolve()}")
        except Exception:
            pass

    return report_path
