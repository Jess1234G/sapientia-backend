# ============================================================
# Sapientia Backend — Imagen de producción
# Python 3.11 slim + dependencias pineadas.
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias del sistema mínimas (curl para healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Primero copiamos requirements para aprovechar la caché de capas
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Código de la aplicación
COPY app ./app
COPY scripts ./scripts

# Usuario no root por seguridad
RUN useradd -m sapientia
USER sapientia

EXPOSE 8000

# Por defecto uvicorn; docker-compose puede sobrescribir command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
