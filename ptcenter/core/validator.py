"""
ptcenter.core.validator
=======================
All input validation functions.  Every external value that will be passed to
a subprocess or used in a URL/file path must pass through one of these before
reaching the caller.
"""

from __future__ import annotations

import ipaddress
import re
import urllib.parse
from pathlib import Path


# Public API
def validate_ip(s: str) -> bool:
    """Accept IPv4, IPv6, or CIDR notation."""
    s = s.strip()
    try:
        ipaddress.ip_network(s, strict=False)
        return True
    except ValueError:
        pass
    try:
        ipaddress.ip_address(s)
        return True
    except ValueError:
        return False


def validate_domain(s: str) -> bool:
    """
    RFC-compliant domain check.  Each label 1-63 chars, total ≤253,
    labels contain only [a-zA-Z0-9-], first/last char not a hyphen.
    Single-label names (e.g. 'localhost') are accepted.
    """
    s = s.strip().lower()
    if not s or len(s) > 253:
        return False
    # Strip trailing dot (FQDN style)
    if s.endswith("."):
        s = s[:-1]
    label_re = re.compile(r"^[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?$")
    parts = s.split(".")
    return all(label_re.match(p) for p in parts)


def validate_url(s: str) -> bool:
    """Must start with http:// or https:// and have a non-empty host."""
    s = s.strip()
    try:
        parsed = urllib.parse.urlparse(s)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def validate_port(s: str | int) -> bool:
    """Integer 1–65535."""
    try:
        port = int(str(s).strip())
        return 1 <= port <= 65535
    except ValueError:
        return False


def validate_target(s: str) -> bool:
    """Accept IP (v4/v6/CIDR) or domain name — union of the above two."""
    s = s.strip()
    return validate_ip(s) or validate_domain(s)


def validate_file_path(s: str, must_exist: bool = True) -> bool:
    """Basic file-path sanity check; optionally verifies the file exists."""
    s = s.strip()
    if not s:
        return False
    p = Path(s)
    if must_exist:
        return p.exists() and p.is_file()
    return True


def validate_cidr(s: str) -> bool:
    """Strict CIDR notation (e.g. 10.0.0.0/8)."""
    try:
        ipaddress.ip_network(s.strip(), strict=True)
        return True
    except ValueError:
        return False


def sanitize_nmap_flags(flags: str) -> list[str]:
    """
    Parse a user-supplied nmap flags string into a list of safe tokens.
    Rejects tokens that look like shell injection: semicolons, pipes,
    backticks, $(...), &&, ||, redirections.
    Returns the cleaned list or raises ValueError on suspicious input.
    """
    import shlex
    dangerous = re.compile(r"[;|`$&<>]")
    tokens = shlex.split(flags)
    for tok in tokens:
        if dangerous.search(tok):
            raise ValueError(f"Potentially dangerous flag token rejected: {tok!r}")
    return tokens


def is_in_scope(target: str, scope: list[str]) -> bool:
    """
    Return True if *target* falls within any network in *scope*.
    Scope entries may be CIDR ranges or individual IPs/domains.
    If scope is empty, every target is considered in-scope.
    """
    if not scope:
        return True
    try:
        target_addr = ipaddress.ip_address(target.strip())
    except ValueError:
        # Domain — treat as in-scope if scope contains the exact domain or a
        # wildcard entry like "*.example.com"
        target_lower = target.strip().lower()
        for entry in scope:
            entry = entry.strip().lower()
            if entry == target_lower:
                return True
            if entry.startswith("*.") and target_lower.endswith(entry[1:]):
                return True
        return False

    for entry in scope:
        try:
            network = ipaddress.ip_network(entry.strip(), strict=False)
            if target_addr in network:
                return True
        except ValueError:
            # Not a network — try exact match
            try:
                if ipaddress.ip_address(entry.strip()) == target_addr:
                    return True
            except ValueError:
                continue
    return False
