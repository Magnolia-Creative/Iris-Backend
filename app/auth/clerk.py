from dataclasses import dataclass
import logging
from typing import Any

from fastapi import HTTPException, Request, WebSocket, status
from jwt import PyJWKClient
from jwt import PyJWKClientError
from jwt import decode as jwt_decode
from jwt import InvalidTokenError

from app.config import settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClerkPrincipal:
    user_id: str
    session_id: str | None
    claims: dict[str, Any]


_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        logger.info("[auth] Initializing Clerk JWKS client url=%s", settings.clerk_jwks_url)
        _jwks_client = PyJWKClient(settings.clerk_jwks_url)
    return _jwks_client


def _extract_bearer_token(value: str | None) -> str | None:
    if not value:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _request_summary(request: Request) -> str:
    return f"{request.method} {request.url.path}"


def _websocket_summary(websocket: WebSocket) -> str:
    return f"WS {websocket.url.path}"


def _principal_from_claims(claims: dict[str, Any]) -> ClerkPrincipal:
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise InvalidTokenError("Missing subject claim.")

    session_id = claims.get("sid")
    if session_id is not None and not isinstance(session_id, str):
        session_id = None

    return ClerkPrincipal(user_id=subject, session_id=session_id, claims=claims)


def verify_clerk_token(token: str) -> ClerkPrincipal:
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        claims = jwt_decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=settings.clerk_issuer,
            options={"require": ["exp", "nbf", "sub"]},
        )
    except (InvalidTokenError, PyJWKClientError) as exc:
        logger.warning(
            "[auth] Clerk token verification failed reason=%s message=%r issuer=%s jwks_url=%s",
            exc.__class__.__name__,
            str(exc),
            settings.clerk_issuer,
            settings.clerk_jwks_url,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Clerk session token.",
        ) from exc

    azp = claims.get("azp")
    if (
        settings.clerk_authorized_parties
        and isinstance(azp, str)
        and azp not in settings.clerk_authorized_parties
    ):
        logger.warning(
            "[auth] Clerk token rejected invalid_azp azp=%r allowed=%s user_id=%r",
            azp,
            settings.clerk_authorized_parties,
            claims.get("sub"),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Clerk authorized party.",
        )

    status_claim = claims.get("sts")
    if status_claim == "pending":
        logger.warning("[auth] Clerk token rejected pending status user_id=%r", claims.get("sub"))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clerk session is pending.",
        )

    principal = _principal_from_claims(claims)
    logger.info(
        "[auth] Clerk token verified user_id=%s session_id=%s azp=%r issuer=%r",
        principal.user_id,
        principal.session_id,
        azp,
        claims.get("iss"),
    )
    return principal


async def require_clerk_user(request: Request) -> ClerkPrincipal:
    auth_header = request.headers.get("Authorization")
    token = _extract_bearer_token(auth_header)
    if token is None:
        logger.warning(
            "[auth] Missing or malformed Authorization header request=%s header_present=%s",
            _request_summary(request),
            auth_header is not None,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Clerk session token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.info("[auth] Authorization header present request=%s", _request_summary(request))
    try:
        return verify_clerk_token(token)
    except HTTPException as exc:
        logger.warning(
            "[auth] Request rejected request=%s status=%s detail=%s",
            _request_summary(request),
            exc.status_code,
            exc.detail,
        )
        raise


def _websocket_token(websocket: WebSocket) -> str | None:
    token = _extract_bearer_token(websocket.headers.get("Authorization"))
    if token is not None:
        return token
    query_token = websocket.query_params.get("token")
    if query_token:
        return query_token
    return None


async def require_clerk_websocket_user(websocket: WebSocket) -> ClerkPrincipal:
    token = _websocket_token(websocket)
    if token is None:
        logger.warning(
            "[auth] Missing WebSocket Clerk token request=%s auth_header_present=%s query_token_present=%s",
            _websocket_summary(websocket),
            websocket.headers.get("Authorization") is not None,
            websocket.query_params.get("token") is not None,
        )
        await websocket.close(code=1008, reason="Missing Clerk session token.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token.")
    try:
        logger.info("[auth] WebSocket Clerk token present request=%s", _websocket_summary(websocket))
        return verify_clerk_token(token)
    except HTTPException as exc:
        logger.warning(
            "[auth] WebSocket rejected request=%s status=%s detail=%s",
            _websocket_summary(websocket),
            exc.status_code,
            exc.detail,
        )
        await websocket.close(code=1008, reason="Invalid Clerk session token.")
        raise


def reset_clerk_jwks_cache() -> None:
    global _jwks_client
    _jwks_client = None
