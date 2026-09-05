"""
security.py — Dependencia FastAPI que valida el token de sesión.

Estrategia en dos niveles:
  1. ID token de Firebase/Google (validado con firebase_admin).
  2. JWT propio (HS256) emitido por el backend tras el login.

La dependencia `get_current_user` inyecta el `uid` autenticado en cada
endpoint protegido.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.services.firebase.auth_service import AuthService, get_auth_service

# Bearer HTTP authentication (Swagger muestra candado en los endpoints)
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    auth: AuthService = Depends(get_auth_service),
) -> str:
    """
    Dependencia de inyección: devuelve el `uid` del usuario autenticado.

    Eleva HTTPException 401 si el token falta o no es válido.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Falta el encabezado Authorization: Bearer <token>",
        )

    uid = auth.validate_token(credentials.credentials)
    if uid is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
        )
    return uid
