import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

# 1. Configurar Pinecone
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "sapientia-pensums")
pc = Pinecone(api_key=api_key)

# 2. Verificar si el índice existe, si no, crearlo
try:
    indices = pc.list_indexes().names()
    print("✅ Conexión a Pinecone exitosa.")
except Exception as e:
    print(f"❌ Error de autenticación: {e}. Verifica tu PINECONE_API_KEY.")
    exit()

if index_name not in indices:
    print(f"🛠️ El índice '{index_name}' no existe. Creándolo...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(5)
    print("✅ Índice creado.")
else:
    print(f"✅ El índice '{index_name}' ya existe.")

index = pc.Index(index_name)

# 3. Cargar modelo de embeddings
print("⏳ Cargando modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Modelo cargado.")

# 4. Datos de los pensums
carreras = {
    "Fisica": ["Electricidad y Magnetismo", "Mecanica Clasica", "Fisica Cuantica", "Teoria Electromagnetica", "Mecanica Estadistica", "Termodinamica"],
    "Matematicas": ["Calculo III", "Ecuaciones Diferenciales", "Variable Compleja", "Analisis I", "Topologia General", "Teoria de Grupos"],
    "Quimica": ["Fisicoquimica I", "Quimica Cuantica", "Quimica Analitica", "Quimica Organica", "Biotecnologia", "Bioquimica"],
    "Ing_Agroindustrial": ["Balance de Materia", "Termodinamica Aplicada", "Operaciones Unitarias", "Diseno de Plantas", "Microbiologia Industrial", "Investigacion de Operaciones"]
}

# 5. Subir datos con IDs únicos y sin caracteres especiales
ids, vectors, metadatos = [], [], []
id_counter = 0

for carrera, materias in carreras.items():
    for materia in materias:
        texto_completo = f"Carrera: {carrera}. Materia: {materia}"
        vector = model.encode(texto_completo).tolist()
        
        # ID limpio (sin tildes, sin espacios, solo ASCII)
        vector_id = f"vec-{id_counter}"
        
        ids.append(vector_id)
        vectors.append(vector)
        metadatos.append({"texto": texto_completo, "carrera": carrera, "materia": materia})
        id_counter += 1

# Subir a Pinecone
index.upsert(vectors=list(zip(ids, vectors, metadatos)))
print(f"✅ ¡Éxito! Se sembraron {len(ids)} materias en Pinecone.")