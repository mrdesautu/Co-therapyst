import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import pandas as pd
import os
import re
import numpy as np

def cargar_embeddings_youtube(embedding_model):
    """Carga los subtemas y embeddings de YouTube solo una vez por sesión."""
    if "youtube_df" not in st.session_state:
        df_videos = pd.read_csv("youtube_psicologia/videos_db.csv")
        st.session_state.youtube_df = df_videos
        st.session_state.youtube_subtemas = df_videos['subtema'].fillna("").tolist()
        st.session_state.youtube_archivos = df_videos['archivo'].fillna("").tolist()
        st.session_state.youtube_embs = [embedding_model.embed_query(subtema) for subtema in st.session_state.youtube_subtemas]
    return st.session_state.youtube_df, st.session_state.youtube_embs, st.session_state.youtube_archivos

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def guardar_historial(historial, nombre_usuario):
    os.makedirs("historiales", exist_ok=True)
    nombre_limpio = re.sub(r'\W+', '_', nombre_usuario)
    nombre_archivo = f"historiales/historial_{nombre_limpio}.txt"
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        for turno in historial:
            f.write(f"Tú: {turno['pregunta']}\n")
            f.write(f"Chatbot: {turno['respuesta']}\n")
            f.write(f"Fecha de sesión: {turno.get('fecha_sesion', 'No disponible')}\n")
            f.write(f"Fuente usada: {turno.get('fuente_usada', 'No disponible')}\n")
            f.write(f"Revisor: {turno.get('revision', 'Sin revisión')}\n")
            f.write("-" * 40 + "\n")
    return nombre_archivo

def chatbot_interface_dashboard():
    st.title("🤖 Chatbot de Caso Clínico")
    st.markdown('<hr style="border-top: 2px solid #A3C9A8; margin-top: 0.5em; margin-bottom: 1em;">', unsafe_allow_html=True)
    st.write("Hazle preguntas al chatbot sobre el caso clínico del paciente o, si lo deseas, incluyendo la base de videos de terapia.")

    nombre_usuario = st.text_input("👤 Ingresa tu nombre (profesional o usuario):")
    if not nombre_usuario:
        st.info("Por favor, ingresa tu nombre para continuar.")
        st.stop()

    openai_key = st.text_input(
        "🔑 Ingresa tu OpenAI API Key para activar el chatbot:",
        type="password",
        help="Tu clave nunca se guarda, solo se usa en esta sesión."
    )
    if not openai_key:
        st.info("Por favor, ingresa tu OpenAI API Key para continuar.")
        st.stop()

    if "historial" not in st.session_state:
        st.session_state.historial = []

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.HttpClient(host="localhost", port=8000)
    vectorstore = Chroma(
        collection_name="teo_sesiones",
        embedding_function=embedding_model,
        client=client
    )
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_key)

    # Prompts predefinidos
    prompt = PromptTemplate.from_template("""
    Eres un psicólogo clínico experto. Responde de forma clara y profesional usando solo la información relevante del caso clínico y, si se selecciona, también de la base de videos de terapia.

    Contexto:
    {context}

    Pregunta:
    {question}

    Respuesta:
    """)
    relevancia_prompt = PromptTemplate.from_template("""
    Dada la siguiente sesión:
    ---
    {contenido}
    ---
    ¿Esta información es útil y relevante para responder la pregunta: "{pregunta}"? 
    Responde solo "Sí" o "No" y, si es "Sí", resume brevemente lo relevante.
    """)
    review_prompt = PromptTemplate.from_template("""
    Actúa como un revisor experto. Evalúa la siguiente respuesta en cuanto a cohesión, lógica y pertinencia respecto a la pregunta y el contexto. 
    Da una breve retroalimentación y una calificación general (Alta, Media, Baja).

    Pregunta del usuario:
    {pregunta}

    Respuesta del chatbot:
    {respuesta}

    Retroalimentación:
    """)

    # Cargar embeddings y archivos de YouTube solo una vez
    df_videos, youtube_embs, youtube_archivos = cargar_embeddings_youtube(embedding_model)

    user_question = st.text_input("Tu pregunta:")

    if user_question:
        with st.spinner("Pensando..."):
            # 1. Buscar en sesiones clínicas
            retrieved_docs = retriever.get_relevant_documents(user_question)
            fechas = [doc.metadata.get('fecha', 'Fecha no disponible') for doc in retrieved_docs]
            fecha_sesion = fechas[0] if fechas else 'Fecha no disponible'

            # Evaluar relevancia de cada sesión
            contexto_relevante = []
            relevancia_chain = LLMChain(llm=llm, prompt=relevancia_prompt)
            for doc in retrieved_docs:
                resultado = relevancia_chain.invoke({"contenido": doc.page_content, "pregunta": user_question})
                texto_resultado = resultado.get('text') or resultado.get('result') or str(resultado)
                if "sí" in texto_resultado.lower():
                    partes = texto_resultado.split(":", 1)
                    resumen = partes[1].strip() if len(partes) > 1 else doc.page_content
                    contexto_relevante.append(resumen)
            if contexto_relevante:
                contexto = "\n".join(contexto_relevante)
                fuente_usada = "Sesiones del caso clínico"
            else:
                # 2. Buscar en videos de YouTube (semántico, embeddings precalculados)
                pregunta_emb = embedding_model.embed_query(user_question)
                similitudes = [cosine_sim(pregunta_emb, emb) for emb in youtube_embs]
                top_n = 3
                top_idx = np.argsort(similitudes)[-top_n:][::-1]
                subtemas_relevantes = df_videos.iloc[top_idx]
                if subtemas_relevantes.empty:
                    contexto = "No se encontró información relevante en las sesiones ni en los videos."
                    fuente_usada = "Sin información relevante"
                else:
                    textos = []
                    archivos_usados = set()
                    # Cache de textos leídos
                    if "youtube_text_cache" not in st.session_state:
                        st.session_state.youtube_text_cache = {}
                    for archivo in subtemas_relevantes['archivo']:
                        if archivo and archivo not in archivos_usados:
                            archivos_usados.add(archivo)
                            if archivo in st.session_state.youtube_text_cache:
                                textos.append(st.session_state.youtube_text_cache[archivo])
                            else:
                                try:
                                    with open(archivo, "r", encoding="utf-8") as f:
                                        texto = f.read()
                                        textos.append(texto)
                                        st.session_state.youtube_text_cache[archivo] = texto
                                except Exception as e:
                                    st.warning(f"No se pudo leer el archivo {archivo}: {e}")
                    contexto = "\n".join(textos)
                    fecha_sesion = "YouTube (subtemas relevantes)"
                    fuente_usada = "YouTube (subtemas relevantes)"

            # Limitar tamaño del contexto para ahorrar tokens
            max_context_chars = 3000
            if len(contexto) > max_context_chars:
                contexto = contexto[:max_context_chars] + "\n...[contexto recortado]..."

            respuesta = llm.invoke(prompt.format(context=contexto, question=user_question)).content

            # Revisor de la respuesta
            review_chain = LLMChain(llm=llm, prompt=review_prompt)
            review_result = review_chain.invoke({"pregunta": user_question, "respuesta": respuesta})
            revision = review_result.get('text') or review_result.get('result') or review_result

            st.session_state.historial.append({
                "pregunta": user_question,
                "respuesta": respuesta,
                "revision": revision,
                "fecha_sesion": fecha_sesion,
                "contexto_usado": contexto,
                "fuente_usada": fuente_usada
            })

            # Guardar historial en archivo
            nombre_archivo = guardar_historial(st.session_state.historial, nombre_usuario)

    # Mostrar historial de conversación (nuevo arriba)
    if st.session_state.historial:
        for turno in reversed(st.session_state.historial):
            with st.chat_message("user"):
                st.markdown(turno['pregunta'])
            with st.chat_message("assistant"):
                st.markdown(turno['respuesta'])
                with st.expander("Ver contexto usado"):
                    st.markdown(turno.get('contexto_usado', 'No disponible'))

    # Botón para descargar historial
    if st.session_state.historial:
        historial_txt = "\n".join(
            f"Tú: {t['pregunta']}\nChatbot: {t['respuesta']}\n" + "-"*40
            for t in st.session_state.historial
        )
        st.download_button("Descargar historial", historial_txt, file_name=nombre_archivo)

if __name__ == "__main__":
    chatbot_interface_dashboard()