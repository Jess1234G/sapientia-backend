"""
endpoints/__init__.py — Router agregador de la API v1.

Incluye los sub-routers de auth, chat, vision, graphs, history y pensum
para que `main.py` solo haga `include_router(api_v1_router)`.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import auth, chat, graphs, history, pensum, vision

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(chat.router, prefix="/chat", tags=["chat"])
router.include_router(vision.router, prefix="/vision", tags=["vision"])
router.include_router(graphs.router, prefix="/graphs", tags=["graphs"])
router.include_router(history.router, prefix="/history", tags=["history"])
router.include_router(pensum.router, prefix="/pensum", tags=["pensum"])
