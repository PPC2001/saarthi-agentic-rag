"""Mock authentication helpers for local/dev/demo environments.

Provides a fixed set of accounts (defined in Settings.mock_accounts) to
simulate Google Sign-In without a real OIDC flow.  Only active when
AUTH_MODE=mock.  Never deployed to production with real data.
"""

from __future__ import annotations

from pydantic import BaseModel


class MockAccount(BaseModel):
    """Single mock user account used in demo/dev mode."""

    id: str  # "employee" | "admin"
    email: str
    name: str
    roles: list[str]
    departments: list[str] = []


# ── Default mock accounts (overridable via Settings.mock_accounts) ─────────
DEFAULT_MOCK_ACCOUNTS: list[MockAccount] = [
    MockAccount(
        id="employee",
        email="pratik@saarthi.com",
        name="Pratik Chavan",
        roles=["employee"],
        departments=["Operations"],
    ),
    MockAccount(
        id="admin",
        email="saarthi.admin@saarthi.com",
        name="Saarthi Admin",
        roles=["admin", "kb_admin", "compliance_reviewer"],
        departments=["Compliance"],
    ),
]


def mock_claims(acct: MockAccount) -> dict:
    """Build a synthetic token-claims dict for the given mock account.

    Mirrors the structure produced by Google's OIDC userinfo response so
    downstream code (upsert_from_claims, session creation, etc.) works
    identically in both auth modes.
    """
    return {
        "sub": f"mock:{acct.id}",
        "email": acct.email,
        "email_verified": True,
        "name": acct.name,
        "picture": None,
        "hd": None,
        "auth_provider": "mock",
        "roles": acct.roles,
        "departments": acct.departments,
    }
