"""
rag_service.py — Recuperación de contexto RAG (pensums vectorizados).

Filtra por `carrera` y `semestre` del usuario y devuelve los fragmentos
más relevantes para inyectar en el prompt de DeepSeek R1.
"""
from __future__ import annotations

import logging

from app.services.rag.embeddings import EmbeddingService, get_embedding_service
from app.services.rag.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres Sapientia, un tutor universitario experto en ciencias duras (Física, Matemáticas, Química e Ingeniería). Tienes acceso al pensum y a la información del usuario.

## Tu misión
- Ayudar a resolver problemas académicos paso a paso con rigor.
- Contextualizar las respuestas con las asignaturas del pensum del estudiante.
- Si el usuario solicita o conviene graficar, DEBES generar un bloque de código Python en tu respuesta siguiendo estas reglas.

## Reglas para generación de gráficos (obligatorio)
Cuando generes código Python, debe cumplir TODO lo siguiente:

1. **Formato de bloque**: siempre entre ```python ... ```.
2. **Gráfico 2D estático (Matplotlib)**: crea una figura usando `matplotlib.pyplot` y guárdala SIEMPRE en el archivo `figura_2d.png`:
   ```python
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   # ... construye tu gráfico 2D ...
   plt.savefig('figura_2d.png', dpi=150, bbox_inches='tight')
   plt.close()
   ```
3. **Gráfico 3D interactivo (Plotly)**: usa `plotly.graph_objects` (o `plotly.express`) y guarda SIEMPRE un archivo `figura_3d.html` con la gráfica 3D interactiva:
   ```python
   import plotly.graph_objects as go
   # ... construye tu gráfica 3D (por ejemplo go.Surface, go.Scatter3d, go.Mesh3d) ...
   fig = go.Figure(data=...)
   fig.update_layout(scene=dict(aspectmode='data'))
   fig.write_html('figura_3d.html', include_plotlyjs='cdn')
   ```
4. **El código debe ser autónomo**: no dependas de archivos externos, genera los datos dentro del propio script (numpy, math, etc.).
5. **Nombres de archivo exactos**: `figura_2d.png` y `figura_3d.html`. Son los que el backend buscará para mostrar la vista 2D y 3D.

## Formato de la respuesta
- Explica el procedimiento matemático paso a paso (puedes usar LaTeX entre $...$).
- Al final, si hay gráfico, incluye el bloque ```python``` completo con las dos figuras.
- Responde en el idioma del usuario (preferentemente español).

## Contexto del usuario
- Carrera: {carrera}
- Semestre: {semestre}

### Pensum de referencia (RAG)
{rag_context}
"""


class RagService:
    """Recupera contexto de pensum desde el vector store."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embeddings: EmbeddingService | None = None,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.embeddings = embeddings or get_embedding_service()

    def query_context(
        self,
        pregunta: str,
        carrera: str,
        semestre: int,
        top_k: int = 5,
    ) -> str:
        """Devuelve texto con los fragmentos relevantes del pensum."""
        vector = self.embeddings.embed_text(pregunta)
        results = self.vector_store.query(
            vector=vector,
            top_k=top_k,
            filter_={"carrera": carrera, "semestre": semestre},
        )
        if not results:
            return ""

        chunks = []
        for idx, hit in enumerate(results, start=1):
            meta = hit.get("metadata", {})
            chunks.append(
                f"[{idx}] {meta.get('asignatura', 'Asignatura')} "
                f"(semestre {meta.get('semestre', '?')}):\n{meta.get('contenido', '')}"
            )
        return "\n\n".join(chunks)

    def build_prompt(
        self,
        pregunta: str,
        carrera: str,
        semestre: int,
        vision_text: str = "",
    ) -> str:
        """Construye el prompt final para DeepSeek R1 con contexto."""
        rag_context = self.query_context(pregunta, carrera, semestre)
        prompt = SYSTEM_PROMPT.format(carrera=carrera, semestre=semestre, rag_context=rag_context)
        if vision_text:
            prompt += (
                "\n\n## Texto extraído de la imagen del estudiante (para resolver):\n"
                f"{vision_text}"
            )
        prompt += f"\n\n## Pregunta del estudiante\n{pregunta}"
        return prompt


def get_rag_service() -> RagService:
    """Dependencia FastAPI: RagService."""
    return RagService()
