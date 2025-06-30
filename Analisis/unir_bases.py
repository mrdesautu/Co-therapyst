import pandas as pd

# Carga los CSV
df_libros = pd.read_csv("libros_psicologia/libros_db.csv")
df_videos = pd.read_csv("youtube_psicologia/videos_db.csv")

# Añade columna fuente
df_libros["fuente"] = "libro"
df_videos["fuente"] = "video"

# Lista de todas las columnas relevantes para la base unificada (incluye ontología)
columnas = [
    "fuente",
    "libro_id", "video_id",
    "titulo", "archivo", "url",
    "fragmento_id", "tema", "tema_id",
    "subtema", "subtema_id", "resumen",
    "intentos_revisor", "calificacion_final", "aprobado",
    "chroma_collection",
    "subtopico", "eje_funcional", "subcategoria_clinica",
    "eje_llm", "subcategoria_llm", "modelo_llm"
]

# Asegura que ambos DataFrames tengan todas las columnas (rellena con vacío si falta)
for col in columnas:
    if col not in df_libros.columns:
        df_libros[col] = ""
    if col not in df_videos.columns:
        df_videos[col] = ""

# Ordena las columnas
df_libros = df_libros[columnas]
df_videos = df_videos[columnas]

# Une ambos
df_unido = pd.concat([df_libros, df_videos], ignore_index=True)
df_unido.to_csv("base_conocimiento_unificada.csv", index=False, encoding="utf-8")
print("✅ Base de conocimiento unificada creada: base_conocimiento_unificada.csv")