"""Test-only admin / admin sign-in.

Off unless DEMO_LOGIN=true, and then only for requests made on this computer.
It opens the seeded demo student, because the web screens are for students only.
No password is stored for it.
"""
import os

DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin"
DEMO_STUDENT = "aarav_sharma"

_LOOPBACK = {"127.0.0.1", "::1"}
_PROXY_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip")
_HOSTING_MARKERS = ("RENDER", "VERCEL", "DYNO", "RAILWAY_ENVIRONMENT", "K_SERVICE")


def demo_login_allowed(enabled: bool, client_host, headers) -> bool:
    if not enabled or any(os.environ.get(name) for name in _HOSTING_MARKERS):
        return False
    if any(headers.get(name) for name in _PROXY_HEADERS):
        return False
    return client_host in _LOOPBACK
