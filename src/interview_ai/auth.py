"""Backend JWT verification for Supabase-issued access tokens.

Verify flow:
- If SUPABASE_JWT_SECRET is set → HS256 (older / self-hosted Supabase)
- Otherwise → RS256 via Supabase's JWKS endpoint (default for hosted projects)

Usage:
    from interview_ai.auth import get_user_id, AuthError

    token = request.headers["Authorization"].removeprefix("Bearer ").strip()
    user_id = get_user_id(token)   # raises AuthError on failure

Environment:
    SUPABASE_URL        — e.g. https://xyzxyz.supabase.co
    SUPABASE_JWT_SECRET — optional, for HS256 (older projects)
"""
from __future__ import annotations

import os
from typing import Any


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status


def _supabase_url() -> str:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not url:
        raise AuthError(
            "SUPABASE_URL is not set. "
            "Add it to .env: SUPABASE_URL=https://<project>.supabase.co"
        )
    return url


def verify_token(token: str) -> dict[str, Any]:
    """Verify a Supabase JWT and return its decoded payload.

    Raises AuthError on any failure (expired, invalid, missing key, etc.).
    """
    try:
        import jwt as _jwt
    except ImportError as exc:
        raise ImportError(
            "Install PyJWT: pip install 'PyJWT[crypto]>=2.8.0'"
        ) from exc

    if not token:
        raise AuthError("No token provided.")

    jwt_secret = os.environ.get("SUPABASE_JWT_SECRET", "")

    try:
        if jwt_secret:
            # HS256 path (older / self-hosted Supabase)
            payload = _jwt.decode(
                token,
                jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
            )
        else:
            # RS256 path — fetch public key from Supabase JWKS
            jwks_url = f"{_supabase_url()}/auth/v1/.well-known/jwks.json"
            jwks_client = _jwt.PyJWKClient(jwks_url, cache_keys=True)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = _jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience="authenticated",
            )
    except _jwt.ExpiredSignatureError:
        raise AuthError("Token has expired.", status=401)
    except _jwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid token: {exc}", status=401)

    return payload  # type: ignore[return-value]


def get_user_id(token: str) -> str:
    """Return the verified Supabase user ID (sub claim).

    Never trust user_id from the request body — always call this instead.
    """
    payload = verify_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise AuthError("Token missing sub claim.")
    return str(user_id)
