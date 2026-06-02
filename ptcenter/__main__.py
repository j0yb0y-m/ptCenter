"""
ptcenter.__main__
=================
Entry point: python -m ptcenter [options]

Non-interactive (CLI) usage examples:
  python -m ptcenter
  python -m ptcenter --no-check
  python -m ptcenter --module scanner --tool nmap --target 10.10.10.1
  python -m ptcenter --module vuln --target CVE-2024-1234
  python -m ptcenter --output /tmp/my_results
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptcenter",
        description="ptCenter v2 — Advanced Python Penetration Testing Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ptcenter                           Launch interactive mode
  python -m ptcenter --no-check               Skip tool health check on startup
  python -m ptcenter --output /tmp/results    Use custom output directory
  python -m ptcenter --module scanner \\
    --tool nmap --target 10.10.10.1           Non-interactive nmap scan
""",
    )

    parser.add_argument(
        "--module",
        choices=["scanner", "osint", "vuln", "exploit", "network",
                 "password", "webapp", "postexploit"],
        help="Module to run non-interactively",
    )
    parser.add_argument(
        "--tool",
        help="Specific tool within the module (e.g. nmap, nikto, sqlmap)",
    )
    parser.add_argument(
        "--target",
        help="Target IP, domain, URL, or CVE ID",
    )
    parser.add_argument(
        "--profile",
        help="Scan profile name / number (for scanner module)",
    )
    parser.add_argument(
        "--output",
        help="Override output directory",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        default=False,
        help="Skip tool health check on startup",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="ptcenter 2.0",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Override output dir before initialising the app
    if args.output:
        import os
        os.environ["OUTPUT_DIR"] = args.output

    try:
        from ptcenter.core.app import PTCenter
        app = PTCenter(args=args)
        app.run()
    except KeyboardInterrupt:
        print("\n[!] Interrupted — exiting.")
        sys.exit(0)
    except Exception as exc:
        print(f"[!] Fatal error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
