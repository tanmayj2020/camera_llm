"""API key + JWT authentication with multi-tenant isolation."""

import hashlib
import logging
import os
import secrets
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TenantContext:
    tenant_id: str
    user_id: str
    access_level: str
    cameras: list[str] = field(default_factory=list)


_DEFAULT_TENANT = TenantContext(
    tenant_id="default", user_id="anonymous", access_level="admin", cameras=[]
)


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


class AuthProvider:
    def __init__(self, persistence_store):
        self._store = persistence_store

    def create_api_key(
        self,
        tenant_id: str,
        user_id: str,
        access_level: str = "read",
        cameras: list[str] | None = None,
    ) -> str:
        raw_key = secrets.token_urlsafe(32)
        hashed = _hash_key(raw_key)
        self._store.set(
            f"auth:apikey:{hashed}",
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "access_level": access_level,
                "cameras": cameras or [],
            },
        )
        return raw_key

    def verify_api_key(self, raw_key: str) -> Optional[TenantContext]:
        hashed = _hash_key(raw_key)
        data = self._store.get(f"auth:apikey:{hashed}")
        if not data:
            return None
        return TenantContext(
            tenant_id=data["tenant_id"],
            user_id=data["user_id"],
            access_level=data["access_level"],
            cameras=data.get("cameras", []),
        )

    def verify_jwt(self, token: str) -> Optional[TenantContext]:
        secret = os.environ.get("JWT_SECRET")
        if not secret:
            logger.warning("JWT_SECRET not set, cannot verify JWT")
            return None
        try:
            import jwt

            payload = jwt.decode(token, secret, algorithms=["HS256"])
            return TenantContext(
                tenant_id=payload["tenant_id"],
                user_id=payload.get("user_id", ""),
                access_level=payload.get("access_level", "read"),
                cameras=payload.get("cameras", []),
            )
        except Exception as exc:
            logger.warning("JWT verification failed: %s", exc)
            return None


def _auth_enabled() -> bool:
    return os.environ.get("AUTH_ENABLED", "").lower() in ("1", "true", "yes")


_auth_provider: Optional[AuthProvider] = None


def set_auth_provider(provider: AuthProvider) -> None:
    global _auth_provider
    _auth_provider = provider


async def get_current_tenant(request=None) -> TenantContext:
    """FastAPI dependency — checks X-API-Key then Bearer token."""
    if not _auth_enabled():
        return _DEFAULT_TENANT

    from fastapi import HTTPException, Request

    if request is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if _auth_provider is None:
        raise HTTPException(status_code=401, detail="Auth not configured")

    api_key = request.headers.get("X-API-Key")
    if api_key:
        ctx = _auth_provider.verify_api_key(api_key)
        if ctx:
            return ctx

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        ctx = _auth_provider.verify_jwt(token)
        if ctx:
            return ctx

    raise HTTPException(status_code=401, detail="Invalid or missing credentials")


def enforce_camera_access(tenant: TenantContext, camera_id: str) -> None:
    if tenant.cameras and camera_id not in tenant.cameras:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=403,
            detail=f"Access denied to camera {camera_id}",
        )
