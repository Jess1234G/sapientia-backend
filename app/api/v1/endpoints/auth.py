"""
auth.py — Autenticación de Sapientia.

POST /auth/google
    Recibe un ID token de Firebase/Google, valida su identidad,
    crea/actualiza el perfil en Firestore y devuelve un JWT
    propio para las peticiones posteriores.

GET /auth/me
    Devuelve el perfil del usuario autenticado.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.security import get_current_user
from app.models.user import User
from app.services.firebase.auth_service import (
    AuthService,
    get_auth_service,
)
from app.services.firebase.firestore_service import (
    FirestoreService,
    get_firestore_service,
)

router = APIRouter()


# ============================================================
# SCHEMAS
# ============================================================

class GoogleAuthRequest(BaseModel):
    """Solicitud de autenticación con Firebase/Google."""

    id_token: str = Field(..., min_length=1)


class AuthResponse(BaseModel):
    """Respuesta emitida después de autenticar al usuario."""

    access_token: str
    token_type: str = "bearer"
    user: User


# ============================================================
# LOGIN GOOGLE / FIREBASE
# ============================================================

@router.post(
    "/google",
    response_model=AuthResponse,
)
async def login_google(
    payload: GoogleAuthRequest,
    auth_service: AuthService = Depends(get_auth_service),
    firestore: FirestoreService = Depends(get_firestore_service),
):
    """
    Valida el ID token de Firebase/Google.

    Flujo:

    Firebase/Google ID token
        ↓
    verify_google_id_token()
        ↓
    claims del usuario
        ↓
    users/{uid} en Firestore
        ↓
    JWT propio de Sapientia
    """

    claims = auth_service.verify_google_id_token(
        payload.id_token
    )

    if not claims:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de Google/Firebase inválido o expirado.",
        )

    uid = claims.get("uid")

    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El token no contiene un UID válido.",
        )

    user_data = await firestore.upsert_user(
        uid=uid,
        email=claims.get("email") or "",
        name=claims.get("name") or "",
        picture=claims.get("picture") or "",
    )

    user = User.model_validate(user_data)

    session_token = auth_service.create_session_token(
        uid
    )

    return AuthResponse(
        access_token=session_token,
        user=user,
    )


# ============================================================
# USUARIO ACTUAL
# ============================================================

@router.get(
    "/me",
    response_model=User,
)
async def get_me(
    uid: str = Depends(get_current_user),
    firestore: FirestoreService = Depends(
        get_firestore_service
    ),
):
    """
    Devuelve el perfil del usuario autenticado.

    El UID no lo proporciona el cliente en el body:
    procede del Bearer token validado por get_current_user().
    """

    user_data = await firestore.get_user(uid)

    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Perfil de usuario no encontrado.",
        )

    return User.model_validate(user_data)