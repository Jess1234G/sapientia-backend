import os
import requests
import hashlib
import secrets
import string
import time
import asyncio
import re
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from firebase_admin import credentials, initialize_app, firestore
from e2b_code_interpreter import AsyncSandbox

load_dotenv()

# =============================================================
# 1. CONFIGURACIÓN DE FIREBASE
# =============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
cred_path_rel = os.getenv("FIREBASE_CREDENTIALS_PATH")

db = None
try:
    if cred_path_rel:
        cred_path = Path(cred_path_rel)
        if not cred_path.is_absolute():
            cred_path = BASE_DIR / cred_path
        if cred_path.exists():
            cred = credentials.Certificate(str(cred_path))
            initialize_app(cred)
            db = firestore.client()
            print("✅ Firebase conectado correctamente.")
        else:
            print(f"⚠️ Firebase no conectado: El archivo {cred_path} no existe.")
    else:
        print("⚠️ Firebase no conectado: Variable FIREBASE_CREDENTIALS_PATH no definida en .env")
except Exception as e:
    print(f"⚠️ Error al conectar Firebase: {e}")


# =============================================================
# 2. SERVICIOS DE IA (DeepSeek, Pinecone, Tavily, E2B)
# =============================================================
# Modelo único para texto y visión
DEEPSEEK_MODEL = "deepseek-v4-pro"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("⚠️ DeepSeek no configurado: falta DEEPSEEK_API_KEY.")

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY or "missing-key",
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

pc = None
index = None
embedding_model = None
tavily_client = None

try:
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if pinecone_key:
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "sapientia-pensums"))
    else:
        print("⚠️ Pinecone no configurado: falta PINECONE_API_KEY.")
except Exception as e:
    print(f"⚠️ Pinecone no disponible: {type(e).__name__}: {e}")

try:
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        tavily_client = TavilyClient(api_key=tavily_key)
    else:
        print("⚠️ Tavily no configurado: falta TAVILY_API_KEY.")
except Exception as e:
    print(f"⚠️ Tavily no disponible: {type(e).__name__}: {e}")

_embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

app = FastAPI(title="Sapientia API - Visión + Gráficos 3D", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "Sapientia API",
        "version": app.version,
        "deepseek_model": DEEPSEEK_MODEL,
        "firebase": db is not None,
        "pinecone": index is not None,
        "tavily": tavily_client is not None,
        "e2b_configured": bool(os.getenv("E2B_API_KEY")),
    }


@app.get("/")
async def root():
    return {
        "name": "Sapientia API",
        "version": app.version,
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================
# 3. ENDPOINT DE TEXTO
# =============================================================
class PreguntaTexto(BaseModel):
    mensaje: str


@app.post("/preguntar-texto")
async def preguntar_texto(request: PreguntaTexto):
    try:
        if not request.mensaje.strip():
            raise HTTPException(
                status_code=400,
                detail="El campo 'mensaje' no puede estar vacío.",
            )

        if not DEEPSEEK_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="Falta DEEPSEEK_API_KEY en .env.",
            )

        contexto_materias = (
            "No hay contexto del plan de estudios disponible en este momento.\n"
        )

        if index is not None:
            try:
                global embedding_model

                if embedding_model is None:
                    embedding_model = SentenceTransformer(_embedding_model_name)

                vector_pregunta = embedding_model.encode(
                    request.mensaje
                ).tolist()

                resultados_rag = index.query(
                    vector=vector_pregunta,
                    top_k=3,
                    include_metadata=True,
                )

                contexto_materias = (
                    "Según tu plan de estudios, esto coincide con:\n"
                )

                for match in resultados_rag.get("matches", []):
                    metadata = match.get("metadata") or {}
                    texto_match = metadata.get("texto")

                    if texto_match:
                        score = float(match.get("score", 0.0))
                        contexto_materias += (
                            f"- {texto_match} "
                            f"(Relevancia: {score:.2f})\n"
                        )

            except Exception as e:
                print(
                    "⚠️ RAG no disponible para esta consulta: "
                    f"{type(e).__name__}: {e}"
                )

        contexto_web = (
            "\nNo se pudo consultar información web actualizada.\n"
        )

        if tavily_client is not None:
            try:
                resultado_web = tavily_client.search(
                    query=request.mensaje,
                    max_results=2,
                )

                resultados = resultado_web.get("results", [])

                if resultados:
                    contexto_web = "\nInformación actualizada de internet:\n"

                    for r in resultados:
                        contexto_web += (
                            f"- {r.get('title', 'Sin título')}: "
                            f"{r.get('content', '')}\n"
                        )
                else:
                    contexto_web = (
                        "\n(No se encontró información actualizada "
                        "relevante en internet).\n"
                    )

            except Exception as e:
                print(
                    "⚠️ Tavily no disponible para esta consulta: "
                    f"{type(e).__name__}: {e}"
                )

        mensaje_completo = f"""
[CONTEXTO UNIVERSITARIO (Tus pensums)]:
{contexto_materias}

[CONTEXTO ACTUALIZADO (Internet)]:
{contexto_web}

[PREGUNTA DEL ESTUDIANTE]:
{request.mensaje}

---

Eres Sapientia, un tutor universitario experto en ciencias duras.
Responde usando el contexto universitario y la información actualizada
proporcionada. Responde en español, con explicaciones claras y rigurosas.
Cuando sea un problema matemático o físico, explica el procedimiento paso a paso.
"""

        respuesta_ia = await asyncio.to_thread(
            deepseek_client.chat.completions.create,
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Eres Sapientia, un tutor de ciencias duras.",
                },
                {
                    "role": "user",
                    "content": mensaje_completo,
                },
            ],
            stream=False,
        )

        if not respuesta_ia.choices:
            raise HTTPException(
                status_code=502,
                detail=(
                    "DeepSeek no devolvió ninguna opción de respuesta. "
                    f"Modelo: {DEEPSEEK_MODEL}"
                ),
            )

        respuesta_texto = respuesta_ia.choices[0].message.content

        if not respuesta_texto:
            finish_reason = getattr(
                respuesta_ia.choices[0],
                "finish_reason",
                None,
            )

            raise HTTPException(
                status_code=502,
                detail=(
                    "DeepSeek respondió sin contenido final. "
                    f"Modelo: {DEEPSEEK_MODEL}; "
                    f"finish_reason: {finish_reason}"
                ),
            )

        return {
            "respuesta": respuesta_texto,
            "modelo_deepseek": DEEPSEEK_MODEL,
            "rag_disponible": index is not None,
            "web_disponible": tavily_client is not None,
        }

    except HTTPException:
        raise

    except Exception as e:
        print(
            "ERROR DETALLADO EN TEXTO: "
            f"{type(e).__name__}: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Error al consultar Sapientia: "
                f"{type(e).__name__}: {e}"
            ),
        )


# =============================================================
# 4. ENDPOINT DE VISIÓN + GRÁFICOS (SimpleTex + DeepSeek + E2B)
# =============================================================
@app.post("/preguntar-vision")
async def preguntar_con_vision(
    mensaje: str = Form(...),
    archivo: UploadFile = File(...),
    generar_grafico: bool = Form(False),
    usuario_id: str = Form("test_user@mail.com"),
):
    """Imagen -> SimpleTex -> LaTeX -> DeepSeek -> E2B -> gráficos 3D."""
    sandbox = None
    html_data = None
    codigo_python = None
    error_graficos = None
    modelo_actual = DEEPSEEK_MODEL

    try:
        if not DEEPSEEK_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="Falta DEEPSEEK_API_KEY en .env.",
            )

        if not mensaje.strip():
            raise HTTPException(
                status_code=400,
                detail="El campo 'mensaje' no puede estar vacío.",
            )

        # 1. Imagen
        contenido_bytes = await archivo.read()
        if not contenido_bytes:
            raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")

        content_type = (archivo.content_type or "application/octet-stream").lower()
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"El archivo debe ser una imagen. Tipo recibido: {content_type}"
            )

        # 2. SimpleTex
        simpletex_app_id = os.getenv("SIMPLETEX_APP_ID")
        simpletex_app_secret = os.getenv("SIMPLETEX_APP_SECRET")
        if not simpletex_app_id or not simpletex_app_secret:
            raise HTTPException(
                status_code=500,
                detail="Faltan SIMPLETEX_APP_ID o SIMPLETEX_APP_SECRET en .env."
            )

        url_simpletex = os.getenv(
            "SIMPLETEX_API_URL",
            "https://server.simpletex.net/api/latex_ocr"
        )

        # 3. Firma APP SimpleTex
        timestamp = str(int(time.time()))
        alphabet = string.ascii_letters + string.digits
        random_str = "".join(secrets.choice(alphabet) for _ in range(16))
        sign_params = {
            "app-id": simpletex_app_id,
            "random-str": random_str,
            "timestamp": timestamp,
        }
        sign_string = "&".join(
            f"{key}={sign_params[key]}" for key in sorted(sign_params)
        )
        signature = hashlib.md5(
            f"{sign_string}&secret={simpletex_app_secret}".encode("utf-8")
        ).hexdigest()

        headers_simpletex = {
            "app-id": simpletex_app_id,
            "random-str": random_str,
            "timestamp": timestamp,
            "sign": signature,
            "Accept": "application/json",
            "User-Agent": "Sapientia/1.0",
        }
        files_simpletex = {
            "file": (
                archivo.filename or "imagen.png",
                contenido_bytes,
                content_type,
            )
        }

        try:
            respuesta_simpletex = await asyncio.to_thread(
                requests.post,
                url_simpletex,
                headers=headers_simpletex,
                files=files_simpletex,
                timeout=(10, 60),
            )
        except requests.exceptions.Timeout as exc:
            raise HTTPException(status_code=504, detail="Tiempo de espera agotado al conectar con SimpleTex.") from exc
        except requests.exceptions.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"No fue posible conectar con SimpleTex: {exc}") from exc

        if respuesta_simpletex.status_code != 200:
            cf_ray = respuesta_simpletex.headers.get("CF-Ray", "no disponible")
            body = respuesta_simpletex.text[:2000]
            print(f"ERROR SIMPLETEX | HTTP={respuesta_simpletex.status_code} | CF-Ray={cf_ray} | BODY={body}")
            raise HTTPException(
                status_code=502,
                detail=f"SimpleTex respondió HTTP {respuesta_simpletex.status_code}: {body}"
            )

        try:
            datos_simpletex = respuesta_simpletex.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="SimpleTex respondió HTTP 200, pero no devolvió JSON válido.") from exc

        if not datos_simpletex.get("status"):
            raise HTTPException(status_code=502, detail=f"SimpleTex rechazó la solicitud: {datos_simpletex}")

        resultado_ocr = datos_simpletex.get("res") or {}
        latex_extraido = resultado_ocr.get("latex", "")
        texto_extraido = resultado_ocr.get("text", "")

        if not latex_extraido and not texto_extraido:
            raise HTTPException(status_code=502, detail="SimpleTex no devolvió texto ni LaTeX.")

        # 4. DeepSeek - PROMPT PARA EXCLUSIVAMENTE GRÁFICO 3D
        prompt_matematico = f"""
[TEXTO EXTRAÍDO POR OCR]:
{texto_extraido}

[LATEX EXTRAÍDO]:
{latex_extraido}

[PREGUNTA DEL USUARIO]:
{mensaje}

[GENERAR GRÁFICO]:
{generar_grafico}

---

Eres Sapientia, un tutor universitario experto en matemáticas, física,
ingeniería y ciencias exactas.

Resuelve el problema paso a paso y en español.

Si GENERAR GRÁFICO es True, incluye un único script de Python ejecutable.
El código debe:
1. Usar Plotly para generar un gráfico 3D interactivo.
2. Guardarlo exactamente como plot_3d.html.
3. No usar fig.show().
4. No solicitar input().
5. No ejecutar pip install.
6. Ser autocontenido.
7. Utilizar datos coherentes con el problema.

Usa fig.write_html("plot_3d.html", include_plotlyjs=True, full_html=True).

Si GENERAR GRÁFICO es True, después de la explicación matemática debes incluir el script completo entre los marcadores exactos <PYTHON_CODE> y </PYTHON_CODE>. No escribas código fuera de esos marcadores. Dentro de los marcadores debe haber únicamente Python ejecutable.
"""

        print(
            "🤖 DeepSeek modelo seleccionado: "
            f"{modelo_actual} | generar_grafico={generar_grafico}"
        )

        respuesta_razonador = await asyncio.to_thread(
            deepseek_client.chat.completions.create,
            model=modelo_actual,
            messages=[
                {"role": "system", "content": "Eres un tutor universitario de ingeniería y ciencias exactas."},
                {"role": "user", "content": prompt_matematico},
            ],
            max_tokens=8192,
            stream=False,
        )
        if not respuesta_razonador.choices:
            raise HTTPException(
                status_code=502,
                detail=(
                    "DeepSeek no devolvió ninguna opción de respuesta. "
                    f"Modelo: {DEEPSEEK_MODEL}"
                )
            )

        mensaje_deepseek = respuesta_razonador.choices[0].message
        respuesta_final = mensaje_deepseek.content

        if not respuesta_final:
            finish_reason = getattr(
                respuesta_razonador.choices[0],
                "finish_reason",
                None
            )
            reasoning_present = bool(
                getattr(
                    mensaje_deepseek,
                    "reasoning_content",
                    None
                )
            )

            raise HTTPException(
                status_code=502,
                detail=(
                    "DeepSeek respondió, pero message.content está vacío. "
                    f"Modelo: {modelo_actual}; "
                    f"finish_reason: {finish_reason}; "
                    f"reasoning_content_present: {reasoning_present}"
                )
            )

        # 5. E2B (SOLO PARA EL GRÁFICO 3D)
        if generar_grafico:
            # ---------------------------------------------------------
            # EXTRAER EL PYTHON DE FORMA ROBUSTA
            # ---------------------------------------------------------
            def extraer_codigo_python(texto: str):
                if not texto:
                    return None

                # 1. Marcadores explícitos.
                match = re.search(
                    r"<PYTHON_CODE>\s*(.*?)\s*</PYTHON_CODE>",
                    texto,
                    re.IGNORECASE | re.DOTALL,
                )
                if match:
                    candidato = match.group(1).strip()
                    if candidato:
                        return candidato

                # 2. Bloque Markdown: ```python ... ``` o ```py ... ```.
                match = re.search(
                    r"```(?:python|py)\s*(.*?)```",
                    texto,
                    re.IGNORECASE | re.DOTALL,
                )
                if match:
                    candidato = match.group(1).strip()
                    if candidato:
                        return candidato

                # 3. Fallback: el modelo a veces entrega el código directamente después de "Código Python:" sin fences.
                lineas = texto.splitlines()

                patrones_import = (
                    re.compile(
                        r"^\s*import\s+"
                        r"(?:numpy|matplotlib|plotly|pandas|scipy|math)"
                        r"\b"
                    ),
                    re.compile(
                        r"^\s*from\s+"
                        r"(?:numpy|matplotlib|plotly|pandas|scipy|math)"
                        r"(?:\.|\b)"
                    ),
                )

                inicio = None
                for indice, linea in enumerate(lineas):
                    if any(p.search(linea) for p in patrones_import):
                        inicio = indice
                        break

                if inicio is not None:
                    candidato = "\n".join(lineas[inicio:]).strip()
                    candidato = re.sub(
                        r"\n```\s*$",
                        "",
                        candidato,
                        flags=re.DOTALL,
                    ).strip()
                    if candidato:
                        return candidato

                return None

            codigo_python = extraer_codigo_python(respuesta_final)

            if not codigo_python:
                error_graficos = (
                    "DeepSeek resolvió el problema, pero no se pudo "
                    "identificar un script Python ejecutable para el gráfico 3D."
                )
            else:
                # Validación local antes de enviar código a E2B.
                try:
                    compile(
                        codigo_python,
                        "<deepseek_generated_code>",
                        "exec",
                    )
                except SyntaxError as exc:
                    codigo_python = None
                    error_graficos = (
                        "DeepSeek generó código Python con un error de "
                        f"sintaxis en la línea {exc.lineno}: {exc.msg}"
                    )

            if codigo_python and not error_graficos:
                e2b_key = os.getenv("E2B_API_KEY")
                if not e2b_key:
                    error_graficos = "E2B_API_KEY no está configurada en .env."
                else:
                    try:
                        # E2B Async
                        sandbox = await AsyncSandbox.create(
                            api_key=e2b_key,
                            timeout=120,
                        )

                        print(
                            "✅ E2B sandbox creado: "
                            f"{getattr(sandbox, 'sandbox_id', 'unknown')}"
                        )

                        # Descubrir el cwd real del contexto Python.
                        cwd_execution = await sandbox.run_code(
                            "import os; print(os.getcwd())",
                            language="python",
                            timeout=10,
                            request_timeout=20,
                        )

                        if cwd_execution.error:
                            raise RuntimeError(
                                "No se pudo determinar el directorio de "
                                f"trabajo de E2B: {cwd_execution.error}"
                            )

                        cwd_lines = (
                            (cwd_execution.text or "")
                            .strip()
                            .splitlines()
                        )
                        cwd = cwd_lines[-1].strip() if cwd_lines else "/home/user"
                        if not cwd.startswith("/"):
                            cwd = "/home/user"

                        print(f"📁 E2B cwd detectado: {cwd}")

                        # Ejecutar el código generado por DeepSeek.
                        execution = await sandbox.run_code(
                            codigo_python,
                            language="python",
                            timeout=60,
                            request_timeout=90,
                        )

                        if execution.error:
                            error_obj = execution.error
                            error_name = getattr(
                                error_obj,
                                "name",
                                "ExecutionError"
                            )
                            error_value = getattr(
                                error_obj,
                                "value",
                                str(error_obj)
                            )
                            error_traceback = getattr(
                                error_obj,
                                "traceback",
                                ""
                            )

                            error_graficos = (
                                f"El código Python falló en E2B ({error_name}): "
                                f"{error_value}"
                            )
                            if error_traceback:
                                error_graficos += f"\nTraceback:\n{error_traceback}"

                            print(f"❌ {error_graficos}")

                        else:
                            print("✅ Código ejecutado correctamente en E2B.")

                            async def read_generated_file(filename: str):
                                candidates = [
                                    f"{cwd}/{filename}",
                                    filename,
                                    f"/home/user/{filename}",
                                ]
                                seen = set()
                                last_error = None
                                for path in candidates:
                                    if path in seen:
                                        continue
                                    seen.add(path)
                                    try:
                                        content = await sandbox.files.read(path)
                                        return path, content
                                    except Exception as exc:
                                        last_error = exc
                                raise FileNotFoundError(
                                    f"No se encontró {filename}. "
                                    f"Rutas comprobadas: {candidates}. "
                                    f"Último error: {last_error}"
                                )

                            # -------------------------------
                            # HTML 3D
                            # -------------------------------
                            try:
                                html_path, html_content = (
                                    await read_generated_file("plot_3d.html")
                                )

                                if isinstance(html_content, str):
                                    html_bytes = html_content.encode("utf-8")
                                elif isinstance(html_content, (bytes, bytearray)):
                                    html_bytes = bytes(html_content)
                                else:
                                    raise TypeError(
                                        "sandbox.files.read() devolvió un tipo "
                                        "no válido para plot_3d.html."
                                    )

                                html_data = base64.b64encode(
                                    html_bytes
                                ).decode("ascii")
                                print(f"✅ plot_3d.html leído desde {html_path}")

                            except Exception as exc:
                                html_error = (
                                    "E2B ejecutó el código, pero no se pudo "
                                    f"leer plot_3d.html: {exc}"
                                )
                                if error_graficos:
                                    error_graficos += f" | {html_error}"
                                else:
                                    error_graficos = html_error
                                print(f"⚠️ {html_error}")

                    except Exception as exc:
                        error_graficos = (
                            "Error controlado en E2B: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        print(f"❌ {error_graficos}")

                    finally:
                        if sandbox is not None:
                            try:
                                await sandbox.kill()
                                print("🧹 E2B sandbox eliminado.")
                            except Exception as close_error:
                                print(
                                    "⚠️ No se pudo eliminar el sandbox E2B: "
                                    f"{close_error}"
                                )

            else:
                error_graficos = (
                    "DeepSeek resolvió el problema, pero no generó "
                    "un bloque de código Python válido para el gráfico 3D."
                )
                print(f"⚠️ {error_graficos}")

        # 6. Firebase
        if db is not None:
            db.collection("usuarios").document(usuario_id).collection("conversaciones").add({
                "pregunta": mensaje,
                "respuesta": respuesta_final,
                "tiene_imagen": True,
                "latex_extraido": latex_extraido,
                "tiene_graficos": html_data is not None,
                "timestamp": firestore.SERVER_TIMESTAMP,
            })
            estado_memoria = "✅ Conversación guardada en Firestore"
        else:
            estado_memoria = "⚠️ Memoria no guardada (Firebase no conectado)"

        return {
            "respuesta": respuesta_final,
            "texto_extraido_ocr": texto_extraido,
            "latex_extraido": latex_extraido,
            "grafico_3d_html_base64": html_data,
            "grafico_3d_html_data_uri": (
                f"data:text/html;base64,{html_data}"
                if html_data
                else None
            ),
            "grafico_3d_generado": html_data is not None,
            "grafico_3d_tipo": "text/html" if html_data else None,
            "error_graficos": error_graficos,
            "codigo_python_generado": codigo_python,
            "modelo_deepseek": modelo_actual,
            "usuario_id": usuario_id,
            "memoria_guardada": estado_memoria,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR DETALLADO EN /preguntar-vision: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error controlado en /preguntar-vision: {type(e).__name__}: {e}"
        )



# =============================================================
# 7. EJECUCIÓN LOCAL
# =============================================================
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
    )