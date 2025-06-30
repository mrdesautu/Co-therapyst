import re
import logging
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from llm_chains import generar_vision_global_final, generar_visiones_generales, extraer_partes_analisis, generar_preguntas_sugeridas
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

client = chromadb.HttpClient(host="localhost", port=8000)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(collection_name="psicologia_clinica_general", embedding_function=embedding_model, client=client)
retriever = vectorstore.as_retriever()

def parse_emociones_pesos_y_frases(texto):
    """
    Extrae emociones, pesos y frases del texto generado por el LLM.
    """
    emociones = {}
    pattern = r"(\w+)\s*:\s*([0-9]*\.?[0-9]+),\s*frases\s*:\s*\"([^\"]+)\""
    for emo, peso, frases in re.findall(pattern, texto, flags=re.IGNORECASE):
        emociones[emo.lower()] = {"peso": float(peso), "frases": frases.strip()}
    return emociones

def clasificar_subtemas(texto, tematica="psicología clínica", llm=None):
    """
    Usa el LLM para clasificar subtemas principales en el texto.
    """
    prompt = (
        f"Lee el siguiente texto y devuelve una lista de los subtemas principales relacionados con '{tematica}'. "
        "Devuelve solo una lista separada por comas, sin explicaciones.\n\n"
        f"Texto:\n{texto}\n\nSubtemas:"
    )
    try:
        response = llm.invoke(prompt)
        # Si response tiene .content, úsalo
        if hasattr(response, "content"):
            subtemas = response.content.strip()
        else:
            subtemas = str(response).strip()
        return [s.strip().lower() for s in subtemas.split(",") if s.strip()]
    except Exception as e:
        logger.warning(f"Error al clasificar subtemas: {e}")
        return []

def get_chroma_collection_name(base, tema, subtema):
    """
    Genera un nombre seguro y único para una colección ChromaDB.
    """
    import hashlib
    def safe_slug(text, maxlen=48):
        text = re.sub(r"[^\w.-]", "_", text.lower())
        return re.sub(r"_+", "_", text).strip("_")[:maxlen]
    def short_hash(text):
        return hashlib.md5(text.encode()).hexdigest()[:8]
    tema_id = int(hashlib.md5(tema.encode()).hexdigest()[:8], 16)
    subtema_id = int(hashlib.md5(subtema.encode()).hexdigest()[:8], 16)
    return f"{safe_slug(base)}_{short_hash(str(tema_id))}_{short_hash(str(subtema_id))}"

def analizar_sesiones(sesiones, analisis_chain, emo_chain, retriever, embedding_model, client, llm):
    """
    Procesa todas las sesiones, extrae análisis, objetivos, avances y emociones.
    Devuelve un DataFrame y una lista de emociones por sesión.
    """
    resultados = []
    emociones_por_sesion = []
    hipotesis, objetivos = None, ["", "", ""]
    acumulados = [0.0, 0.0, 0.0]
    total_sesiones = len(sesiones)
    max_avance_sesion = (10 / total_sesiones) + 1

    for id_, texto, meta in sorted(sesiones, key=lambda x: x[2].get("n_sesion", 0)):
        n, fecha = meta["n_sesion"], meta["fecha"]

        # Buscar contexto relevante
        subtemas = clasificar_subtemas(texto, llm=llm)
        if subtemas:
            subtema = subtemas[0]
            collection_name = get_chroma_collection_name("youtube_terapia", "psicología clínica", subtema)
            try:
                vectorstore = Chroma(collection_name=collection_name, embedding_function=embedding_model, client=client)
                retriever_subtema = vectorstore.as_retriever()
                docs = retriever_subtema.invoke(texto)
                contexto_txt = "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                logger.warning(f"No se pudo acceder a '{collection_name}', usando colección general. Error: {e}")
                contexto_txt = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(texto)])
        else:
            contexto_txt = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(texto)])

        # Primera sesión: hipótesis y objetivos
        if n == 1:
            hipotesis = analisis_chain.invoke({
                "session": texto,
                "context": contexto_txt,
                "question": "Formula una hipótesis del motivo de consulta desde la psicología positiva."
            }).content
            objetivos_texto = analisis_chain.invoke({
                "session": texto,
                "context": contexto_txt,
                "question": "Establece 3 objetivos centrales para esta terapia desde la psicología positiva."
            }).content
            objetivos = [line.strip("-•– ") for line in objetivos_texto.split("\n") if line.strip()][:3]
            while len(objetivos) < 3:
                objetivos.append("No especificado")
            avances = [0.0, 0.0, 0.0]
            comentario = "Primera sesión: se formula hipótesis y se establecen los objetivos terapéuticos."
        else:
            avances = []
            for i, objetivo in enumerate(objetivos):
                toco = analisis_chain.invoke({
                    "session": texto,
                    "context": contexto_txt,
                    "question": f"¿Se trabajó activamente el objetivo '{objetivo}' en esta sesión? sí/no"
                }).content
                if "no" in toco.lower():
                    avance = 0.0
                else:
                    evaluacion = analisis_chain.invoke({
                        "session": texto,
                        "context": contexto_txt,
                        "question": f"¿Cuánto se avanzó hacia el objetivo '{objetivo}' en esta sesión? Responde solo con un número entre 0 y 10."
                    }).content
                    match = re.search(r"\d{1,2}(?:\.\d{1,2})?", evaluacion)
                    avance = float(match.group()) if match else 0.0
                    avance = min(avance, max_avance_sesion)
                acumulados[i] = min(acumulados[i] + avance, 10.0)
                avances.append(acumulados[i])
            comentario = analisis_chain.invoke({
                "session": texto,
                "context": contexto_txt,
                "question": "Resume brevemente el estado terapéutico del paciente en esta sesión."
            }).content

        # Emociones
        emo_text = emo_chain.invoke({
            "context": contexto_txt,
            "session": texto
        }).content
        emo_dict = parse_emociones_pesos_y_frases(emo_text)
        total_peso = sum([v["peso"] for v in emo_dict.values()])
        if total_peso > 0:
            for v in emo_dict.values():
                v["peso"] /= total_peso

        # Agrega emociones a la lista para expandir luego
        emociones_por_sesion.append({
            "n_sesion": n,
            "emociones": emo_dict
        })

        # 1. Genera las visiones generales para la sesión (esto depende de tu pipeline)
        visiones_generales = generar_visiones_generales(contexto_txt, texto)  # dict: {perspectiva: vision}

        # 2. Genera la visión global integradora para esta sesión
        vision_global_final = generar_vision_global_final(visiones_generales)

        # 3. Guarda cada visión general y la visión global en el resultado de la sesión
        fila_resultado = {
            "n_sesion": n,
            "fecha": fecha,
            "hipotesis_motivo_consulta": hipotesis,
            "objetivo_1": objetivos[0],
            "objetivo_2": objetivos[1],
            "objetivo_3": objetivos[2],
            "avance_objetivo_1": avances[0],
            "avance_objetivo_2": avances[1],
            "avance_objetivo_3": avances[2],
            "comentario_terapeutico": comentario,
        }
        # Agrega cada visión general como columna
        for perspectiva, vision in visiones_generales.items():
            col_base = perspectiva.lower().replace(' ', '_')
            # Guarda el texto completo de la visión (opcional)
            fila_resultado[f"vision_{col_base}"] = vision
            # Desglosa hipótesis, tratamiento y avance
            hipotesis, tratamiento, avance = extraer_partes_analisis(vision)
            fila_resultado[f"hipotesis_{col_base}"] = hipotesis
            fila_resultado[f"tratamiento_{col_base}"] = tratamiento
            fila_resultado[f"avance_{col_base}"] = avance
        # Agrega la visión global integradora
        fila_resultado["vision_global_final"] = vision_global_final

        # Preguntas sugeridas
        preguntas_sugeridas = generar_preguntas_sugeridas(contexto_txt, texto)
        for perspectiva, preguntas in preguntas_sugeridas.items():
            col_base = perspectiva.lower().replace(' ', '_')
            fila_resultado[f"preguntas_sugeridas_{col_base}"] = preguntas

        resultados.append(fila_resultado)
    return pd.DataFrame(resultados), emociones_por_sesion

def expandir_emociones(df_avance, emociones_por_sesion):
    """
    Expande las emociones detectadas en columnas separadas para el DataFrame final.
    """
    all_emociones = sorted({emo for e in emociones_por_sesion for emo in e["emociones"].keys()})
    final_rows = []
    for _, row in df_avance.iterrows():
        n = row["n_sesion"]
        emo_dict = next((e["emociones"] for e in emociones_por_sesion if e["n_sesion"] == n), {})
        fila = row.to_dict()
        for emo in all_emociones:
            fila[f"peso_{emo}"] = emo_dict.get(emo, {}).get("peso", 0.0)
            fila[f"frases_{emo}"] = emo_dict.get(emo, {}).get("frases", "")
        final_rows.append(fila)
    return pd.DataFrame(final_rows), all_emociones

def leer_sesiones_chromadb(paciente_id="teo"):
    """
    Lee las sesiones de un paciente desde ChromaDB y las retorna como lista de tuplas (id, texto, meta).
    """
    import chromadb
    client = chromadb.HttpClient(host="localhost", port=8000)
    collection_name = f"{paciente_id}_sesiones"
    vectorstore = Chroma(
        collection_name=collection_name,
        client=client
    )
    data = vectorstore.get()
    sesiones = []
    for doc_id, texto, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        meta = meta or {}
        sesiones.append((doc_id, texto, {
            "n_sesion": meta.get("n_sesion", doc_id),
            "fecha": meta.get("fecha", ""),
            "paciente": paciente_id
        }))
    sesiones.sort(key=lambda x: int(x[2]["n_sesion"]) if str(x[2]["n_sesion"]).isdigit() else x[2]["n_sesion"])
    return sesiones

if __name__ == "__main__":
    try:
        paciente_id = "teo"  # O el paciente que desees analizar
        sesiones = leer_sesiones_chromadb(paciente_id)
        print(f"Se encontraron {len(sesiones)} sesiones para el paciente '{paciente_id}'.")
        from llm_chains import analisis_chain, emo_chain, embedding_model, llm
        df_avance, emociones_por_sesion = analizar_sesiones(
            sesiones, analisis_chain, emo_chain, retriever, embedding_model, client, llm
        )
        df_final, all_emociones = expandir_emociones(df_avance, emociones_por_sesion)
        print(df_final)
        df_final.to_csv("resultados_transversales.csv", index=False)
        print("CSV guardado como resultados_transversales.csv")
    except Exception as e:
        logger.error(f"Error en el procesamiento: {e}")