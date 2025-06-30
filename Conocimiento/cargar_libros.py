"""
Script para procesar libros clínicos (PDF, TXT, DOCX, EPUB):

- Extrae el texto completo de cada libro nuevo en la carpeta 'libros_nuevos'.
- Detecta posibles subtítulos clínicos relevantes usando reglas, una lista de exclusión y validación con OpenAI (usando como filtro las subcategorías clínicas del archivo 'ejes_clinicos.csv').
- Divide el texto en fragmentos según los subtítulos detectados.
- Resume cada fragmento usando un modelo LLM (OpenAI).
- Limpia y normaliza los textos y subtemas.
- Valida que cada fragmento sea clínicamente relevante (herramienta, técnica o recurso) usando LLM.
- Califica y revisa la calidad del resumen y subtema con LLM, repitiendo hasta obtener una calificación suficiente.
- Clasifica ontológicamente cada fragmento: 
    - Asigna subtema y subtema_id.
    - Clasifica funcionalmente usando similitud semántica con las subcategorías del CSV.
    - Clasifica también con LLM para eje y subcategoría clínica.
- Guarda todos los fragmentos válidos en 'libros_psicologia/libros_db.csv', agregando columnas para trazabilidad, ontología y colección ChromaDB.
- Añade los fragmentos a ChromaDB, usando nombres de colección normalizados y metadatos enriquecidos.
- Mantiene un log de procesamiento con métricas de éxito, fallos y uso de tokens.
- Usa cache persistente para acelerar la validación de subtítulos clínicos.
- Evita reprocesar libros ya cargados.
"""

import os
import sys
import re
import hashlib
import unicodedata
import pickle

# Terceros
import pandas as pd
import openai
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from docx import Document
from ebooklib import epub

# Propios/modelos
from sentence_transformers import SentenceTransformer, util

#
# --- Configuración global de modelos y clientes ---
client = openai.OpenAI()  # Usará la variable de entorno OPENAI_API_KEY
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

EJES_CLINICOS_PATH = "ejes_clinicos.csv"
EJES_DF = pd.read_csv(EJES_CLINICOS_PATH)

# Modelo para normalización de subtemas
modelo_subtemas = SentenceTransformer("all-MiniLM-L6-v2")

# --- Normalización de subtopicos ---
def normalizar_texto(texto):
    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.replace('\n', ' ').replace('\r', ' ')
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.strip()
    texto = ''.join(c for c in texto if c.isprintable())
    return texto

def normalizar_subtopico(subtopico):
    subtopico = normalizar_texto(subtopico)
    emb_sub = modelo_subtemas.encode(subtopico, convert_to_tensor=True)
    emb_lista = modelo_subtemas.encode(EJES_DF["subcategoria"].tolist(), convert_to_tensor=True)
    sim = util.cos_sim(emb_sub, emb_lista)[0]
    best_idx = int(sim.argmax())
    if float(sim[best_idx]) >= 0.6:
        return EJES_DF.iloc[best_idx]["eje"], EJES_DF.iloc[best_idx]["subcategoria"]
    return "Sin clasificar", subtopico

# --- Configuración ---
CARPETA = "libros_psicologia"
DB_PATH = os.path.join(CARPETA, "libros_db.csv")
CACHE_PATH = "subtitulos_clinicos_cache.pkl"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Configuración de log de procesamiento ---
LOG_PATH = os.path.join(CARPETA, "procesamiento_log.csv")
LOG_COLUMNS = ["archivo", "fragmentos_totales", "fragmentos_validos", "tokens_usados", "modelo", "exitosos", "fallidos"]

SUBTITULOS_IGNORAR = [
    "índice", "referencias", "bibliografía", "prólogo", "agradecimientos", "introducción",
    "apéndice", "glosario", "anexos", "dedicatoria", "créditos", "copyright",
    "web", "autores", "grupo de psicología", "presentación", "coordinadora", "editorial",
    "contacto", "sobre el autor", "acerca de", "biografía"
]

def es_titulo_ignorable(titulo):
    t = normalizar_texto(titulo).lower()
    return any(pal in t for pal in SUBTITULOS_IGNORAR)

# Cache global para validación de subtítulos clínicos
SUBTITULO_CLINICO_CACHE = {}

def guardar_cache():
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(SUBTITULO_CLINICO_CACHE, f)

def cargar_cache():
    global SUBTITULO_CLINICO_CACHE
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            SUBTITULO_CLINICO_CACHE = pickle.load(f)

def safe_slug(text, maxlen=48):
    text = text.lower()
    text = re.sub(r"[^\w.-]", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text[:maxlen]

def short_hash(text):
    return hashlib.md5(str(text).encode()).hexdigest()[:8]

def libro_id_numeric(text):
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

def subtema_id_numeric(text):
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

def extraer_texto_libro(filepath):
    try:
        if filepath.lower().endswith(".pdf"):
            import PyPDF2
            texto = ""
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    texto += page.extract_text() or ""
            return texto
        elif filepath.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif filepath.lower().endswith(".docx"):
            doc = Document(filepath)
            texto = "\n".join([p.text for p in doc.paragraphs])
            return texto
        elif filepath.lower().endswith(".epub"):
            book = epub.read_epub(filepath)
            texto = ""
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    contenido = item.get_content().decode("utf-8")
                    texto += re.sub('<[^<]+?>', '', contenido) + "\n"
            return texto
        else:
            raise ValueError("Formato de libro no soportado. Usa PDF, TXT, DOCX o EPUB.")
    except Exception as e:
        print(f"❌ Error al extraer texto de {filepath}: {e}")
        return ""

def limpiar_texto(texto):
    return normalizar_texto(texto)

def limpiar_con_llm(texto, tipo="subtema o resumen"):
    prompt = (
        f"Limpia y mejora el siguiente {tipo} para que sea claro, profesional y sin errores. "
        f"Corrige ortografía, gramática y elimina inconsistencias. Devuelve solo el texto limpio:\n\n"
        f"{texto}\n\nTexto limpio:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error limpiando con LLM: {e}")
        return limpiar_texto(texto)

def resumir_texto(texto, tematica="psicología clínica"):
    prompt = (
        f"Resume el siguiente texto de un libro de {tematica} en 3-4 frases claras y concisas:\n\n"
        f"{texto}\n\nResumen:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error al resumir: {e}")
        return texto[:1000]

def es_subtema_clinico(subtitulo):
    clave = normalizar_texto(subtitulo).lower()
    # 1. Verifica coincidencia con subcategorías del CSV (match exacto o parcial)
    subcats = [str(s).strip().lower() for s in EJES_DF["subcategoria"].unique()]
    if any(clave in subcat or subcat in clave for subcat in subcats):
        SUBTITULO_CLINICO_CACHE[clave] = True
        return True
    # 2. Si no coincide, pregunta al LLM usando las categorías como filtro
    if clave in SUBTITULO_CLINICO_CACHE:
        return SUBTITULO_CLINICO_CACHE[clave]
    opciones = "\n".join(f"- {row['subcategoria']}" for _, row in EJES_DF.iterrows())
    prompt = (
        "Dada la siguiente lista de subcategorías clínicas:\n"
        f"{opciones}\n\n"
        f"¿El siguiente título de sección corresponde a alguna de estas subcategorías clínicas?\n"
        f"Responde solo 'Sí' o 'No'.\n\n"
        f"Título: {subtitulo}\nRespuesta:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        resp = response.choices[0].message.content or ""
        es_clinico = "sí" in resp.lower()
        SUBTITULO_CLINICO_CACHE[clave] = es_clinico
        return es_clinico
    except Exception as e:
        print(f"Error validando subtema clínico: {e}")
        SUBTITULO_CLINICO_CACHE[clave] = False
        return False

def es_fragmento_clinico(resumen):
    prompt = (
        "¿El siguiente texto describe una técnica, recurso o herramienta aplicable en la clínica psicológica? "
        "Responde solo 'Sí' o 'No'.\n\n"
        f"Texto: {resumen}\nRespuesta:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        resp = response.choices[0].message.content or ""
        return "sí" in resp.lower()
    except Exception as e:
        print(f"Error validando fragmento clínico: {e}")
        return False

def extraer_subtitulos(texto):
    subtitulos = []
    for match in re.finditer(r'^(?P<titulo>[A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ ]{2,40})(?=\n|:)', texto, re.MULTILINE):
        sub = match.group('titulo').strip()
        if not es_titulo_ignorable(sub):
            if es_subtema_clinico(sub):
                subtitulos.append((match.start(), sub))
    return subtitulos

def chunk_por_subtitulos(texto, subtitulos):
    chunks = []
    for i, (start, sub) in enumerate(subtitulos):
        end = subtitulos[i + 1][0] if i + 1 < len(subtitulos) else len(texto)
        fragmento = texto[start:end]
        chunks.append({'text': fragmento, 'start': start})
    return chunks

def asignar_subtema_a_fragmento(fragmento, subtitulos):
    pos = fragmento['start']
    anterior = None
    for start, sub in subtitulos:
        if start <= pos:
            anterior = sub
        else:
            break
    return anterior or "sin subtema"

def clasificar_eje_clinico(subtopico):
    subtopico_lower = subtopico.strip().lower()
    for _, row in EJES_DF.iterrows():
        if subtopico_lower in row["subcategoria"].strip().lower():
            return row["eje"], row["subcategoria"]
    return "Sin clasificar", subtopico

def clasificar_eje_llm(texto, ejes_df):
    # Construye el prompt con los ejes y subcategorías
    opciones = "\n".join(
        f"{row['eje']} > {row['subcategoria']}" for _, row in ejes_df.iterrows()
    )
    prompt = (
        "Según la siguiente lista de ejes y subcategorías clínicas, "
        "indica a cuál pertenece el siguiente texto. "
        "Devuelve solo el nombre exacto del eje y subcategoría, separados por ' > '.\n\n"
        f"Opciones:\n{opciones}\n\n"
        f"Texto:\n{texto}\n\n"
        "Respuesta:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        respuesta = response.choices[0].message.content.strip()
        if ">" in respuesta:
            eje, subcat = [x.strip() for x in respuesta.split(">", 1)]
            return eje, subcat
        else:
            return "Sin clasificar", "Sin clasificar"
    except Exception as e:
        print(f"Error clasificando eje clínico: {e}")
        return "Sin clasificar", "Sin clasificar"

def procesar_libro(filepath):
    print(f"Procesando: {filepath}")
    texto = extraer_texto_libro(filepath)
    print(f"Longitud del texto extraído: {len(texto)}")
    if not texto.strip():
        print(f"⚠️ Libro vacío o no se pudo extraer texto: {filepath}")
        return []
    subtitulos = extraer_subtitulos(texto)
    print(f"Subtítulos detectados: {subtitulos}")
    chunks = chunk_por_subtitulos(texto, subtitulos)
    libro_title = os.path.splitext(os.path.basename(filepath))[0]
    libro_id = libro_id_numeric(libro_title)
    registros = []
    for i, c in enumerate(chunks):
        resumen = resumir_texto(c['text'])
        subtema = asignar_subtema_a_fragmento(c, subtitulos)
        intento = 0
        max_intentos = 3
        calificacion = 0
        resumen_limpio = ""
        subtema_limpio = ""
        while calificacion < 7 and intento < max_intentos:
            resumen_limpio = limpiar_texto(resumir_texto(c['text']))
            subtema_limpio = limpiar_texto(asignar_subtema_a_fragmento(c, subtitulos))
            if not subtema_limpio or subtema_limpio.lower() == "sin subtema":
                try:
                    infer_prompt = (
                        f"Resume el siguiente texto en 2-5 palabras que describan el concepto clínico principal. "
                        f"Devuelve solo el título sugerido:\n\n{resumen_limpio}\n\nSubtema:"
                    )
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": infer_prompt}],
                        max_tokens=20,
                        temperature=0.5,
                    )
                    subtema_limpio = response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"⚠️ No se pudo inferir subtema desde resumen: {e}")
                    subtema_limpio = "subtema inferido"
            intento += 1
            revision_prompt = (
                f"Evalúa la calidad del siguiente resumen clínico y su subtema. "
                f"Da una calificación del 1 al 10, y explica por qué.\n\n"
                f"Resumen: {resumen_limpio}\nSubtema: {subtema_limpio}\n\n"
                f"Devuelve solo el número entero de calificación:"
            )
            try:
                revision_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": revision_prompt}],
                    max_tokens=10,
                    temperature=0.3,
                )
                calificacion_texto = revision_response.choices[0].message.content.strip()
                matches = re.findall(r"\d+", calificacion_texto or "")
                calificacion = int(matches[0]) if matches else 0
            except Exception as e:
                print(f"⚠️ No se pudo obtener calificación del revisor: {e}")
                calificacion = 0
        if not resumen_limpio:
            continue  # Salta fragmentos vacíos
        subtopico_detectado = subtema_limpio  # guardamos lo que dice el LLM originalmente
        # --- FILTRO CLÍNICO ---
        if (
            not resumen_limpio
            or not subtema_limpio
            or subtema_limpio.lower() == "sin subtema"
            or es_titulo_ignorable(subtema_limpio)
            or len(resumen_limpio) < 60
            or not es_fragmento_clinico(resumen_limpio)
        ):
            continue  # Salta fragmentos vacíos o irrelevantes
        subtema_id = subtema_id_numeric(subtema_limpio)
        # --- Usar normalizar_subtopico en vez de clasificar_eje_clinico ---
        eje_funcional, subcategoria_clinica = normalizar_subtopico(subtema_limpio)
        # Genera el nombre de la colección ChromaDB
        subtema_slug = safe_slug(str(subtema_limpio))
        chroma_collection = f"libros_{short_hash(str(libro_id))}_{normalizar_nombre_chroma(subtema_slug)}"
        eje_llm, subcat_llm = clasificar_eje_llm(resumen_limpio, EJES_DF)
        registros.append({
            "libro_id": libro_id,
            "titulo": libro_title,
            "archivo": filepath,
            "fragmento_id": i,
            "subtema": subtema_limpio,
            "subtema_id": subtema_id,
            "resumen": resumen_limpio[:400],
            "intentos_revisor": intento,
            "calificacion_final": calificacion,
            "aprobado": calificacion >= 7,
            "chroma_collection": chroma_collection,
            "subtopico": subtopico_detectado,
            "eje_funcional": eje_funcional,
            "subcategoria_clinica": subcategoria_clinica,
            "eje_llm": eje_llm,
            "subcategoria_llm": subcat_llm,
        })
    return registros

def normalizar_nombre_chroma(nombre):
    # Elimina acentos y caracteres especiales, deja solo a-zA-Z0-9._-
    nombre = unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode('ascii')
    nombre = re.sub(r'[^a-zA-Z0-9._-]', '_', nombre)
    nombre = re.sub(r'_+', '_', nombre)
    nombre = nombre.strip('_-.')
    # Chroma requiere mínimo 3 caracteres
    if len(nombre) < 3:
        nombre = f"col_{nombre}"
    return nombre[:64]  # Limita el largo si quieres

def main(libros_paths):
    os.makedirs(CARPETA, exist_ok=True)
    cargar_cache()
    # Verifica si el archivo existe y no está vacío
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
        df_old = pd.read_csv(DB_PATH)
        libros_existentes = set(df_old['archivo'].unique())
    else:
        df_old = None
        libros_existentes = set()

    # Evitar repetir libros ya procesados
    libros_nuevos = [p for p in libros_paths if p not in libros_existentes]
    if not libros_nuevos:
        print("No hay libros nuevos para procesar.")
        return

    registros_totales = []
    for path in libros_nuevos:
        registros = procesar_libro(path)
        registros_totales.extend(registros)
    if not registros_totales:
        print("No se encontraron fragmentos válidos en los libros nuevos.")
        return
    df = pd.DataFrame(registros_totales)

    if df_old is not None:
        df = pd.concat([df_old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["libro_id", "fragmento_id", "subtema_id"])
    df.to_csv(DB_PATH, index=False, encoding="utf-8")
    print(f"✅ Base de datos de libros actualizada en: {DB_PATH}")

    guardar_cache()

    # Añadir fragmentos a ChromaDB
    if not df.empty and 'archivo' in df.columns:
        nuevos_fragmentos = df[df['archivo'].isin(libros_nuevos)]
        for _, row in nuevos_fragmentos.iterrows():
            # Usar slug del subtema para mayor trazabilidad
            subtema_limpio = safe_slug(str(row['subtema']))
            collection_name = f"libros_{short_hash(str(row['libro_id']))}_{normalizar_nombre_chroma(subtema_limpio)}"
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                client=chroma_client
            )
            vectorstore.add_texts(
                [row["resumen"]],
                metadatas=[{
                    "source": row["archivo"],
                    "subtema": row["subtema"],
                    "subtopico": row["subtopico"],
                    "eje": row["eje_funcional"],
                    "subcategoria": row["subcategoria_clinica"]
                }]
            )
            print(f"Fragmento agregado a ChromaDB en colección '{collection_name}'.")

    # --- LOG DE PROCESAMIENTO ---
    fragmentos_totales = len(registros_totales)
    fragmentos_validos = len(df[df['archivo'].isin(libros_nuevos)])
    # Permitir logging de tokens usados aunque OpenAI API no devuelva uso
    tokens_usados = getattr(getattr(client, "usage", None), "total_tokens", 0)
    modelo = "gpt-3.5-turbo"
    exitosos = fragmentos_validos
    fallidos = fragmentos_totales - fragmentos_validos
    log_row = pd.DataFrame([{
        "archivo": libros_nuevos[0] if len(libros_nuevos) == 1 else "varios",
        "fragmentos_totales": fragmentos_totales,
        "fragmentos_validos": fragmentos_validos,
        "tokens_usados": tokens_usados,
        "modelo": modelo,
        "exitosos": exitosos,
        "fallidos": fallidos
    }])
    if os.path.exists(LOG_PATH):
        pd.concat([pd.read_csv(LOG_PATH), log_row], ignore_index=True).to_csv(LOG_PATH, index=False)
    else:
        log_row.to_csv(LOG_PATH, index=False)
    print("📝 Log de procesamiento actualizado.")

if __name__ == "__main__":
    CARPETA_NUEVOS = "libros_nuevos"
    os.makedirs(CARPETA_NUEVOS, exist_ok=True)
    extensiones = (".pdf", ".txt", ".docx", ".epub")
    libros_paths = [
        os.path.join(CARPETA_NUEVOS, f)
        for f in os.listdir(CARPETA_NUEVOS)
        if f.lower().endswith(extensiones)
    ]
    if not libros_paths:
        print("No se encontraron libros nuevos en la carpeta 'libros_nuevos'.")
        sys.exit(0)
    main(libros_paths)