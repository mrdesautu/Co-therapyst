import os
import glob
import subprocess
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from dotenv import load_dotenv
load_dotenv()

from Analisis.llm_chains import resumen_chain
from Herramientas.utils import (
    get_emocion_color, avances_a_porcentaje, graficar_emociones_por_sesion,
    graficar_emociones_detalle, EMOCION_COLORES, PASTEL_COLORS
)
from Herramientas.graficos import graficar_progreso
from Herramientas.generar_pdf import generar_pdf

st.set_page_config(page_title="Dashboard Terapéutico", layout="wide", page_icon="🧑‍⚕️")

# --- Estilos CSS personalizados ---
st.markdown("""
    <style>
    body { background-color: #f7f6f2; }
    .stApp { background: linear-gradient(120deg, #f7f6f2 0%, #e2f0cb 100%); }
    .block-container {
        background-color: #ffffffcc;
        border-radius: 18px;
        padding: 2rem 2rem 1rem 2rem;
        box-shadow: 0 4px 24px 0 rgba(163,201,168,0.12);
    }
    .stButton>button {
        background-color: #A3C9A8;
        color: #333;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #B5EAD7;
        color: #222;
    }
    .stRadio>div>label {
        color: #6B9080 !important;
        font-weight: 600;
    }
    .stSidebar { background: #e2f0cb; }
    </style>
""", unsafe_allow_html=True)

# --- Árbol de conocimiento (YouTube) usando CSV ---
def mostrar_arbol_conocimiento_desde_csv(csv_path):
    if not os.path.exists(csv_path):
        st.warning("No se encontró la base de datos de videos.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        st.info("No hay videos cargados aún.")
        return
    for tema in df["tema"].unique():
        with st.expander(tema):
            df_tema = df[df["tema"] == tema]
            for subtema in df_tema["subtema"].unique():
                st.markdown(f"### {subtema.capitalize()}")
                df_sub = df_tema[df_tema["subtema"] == subtema]
                for _, row in df_sub.iterrows():
                    st.markdown(f"**{row['titulo']}**")
                    st.markdown(f"- [Ver en YouTube]({row['url']})")
                    st.markdown(f"_Resumen:_ {row['resumen']}")
                st.markdown("---")


# --- Botón para ejecutar el análisis y actualizar los datos ---
st.sidebar.header("Actualizar análisis")
if st.sidebar.button("Ejecutar análisis y actualizar"):
    try:
        import chromadb
        from langchain_community.vectorstores import Chroma
        from llm_chains import analisis_chain, resumen_chain, emo_chain, embedding_model, ll
        from procesamiento import analizar_sesiones, expandir_emociones
        from graficos import graficar_emociones

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        client = chromadb.HttpClient(host="localhost", port=8000)
        vectorstore = Chroma(
            collection_name=f"{st.session_state.get('paciente_id', 'teo')}_sesiones",
            embedding_function=embedding_model,
            client=client
        )
        data = vectorstore.get()
        sesiones = list(zip(data["ids"], data["documents"], data["metadatas"]))
        retriever = vectorstore.as_retriever()
        df_avance, emociones_por_sesion = analizar_sesiones(
            sesiones, analisis_chain, emo_chain, retriever, embedding_model, client, ll
        )
        df_avance.to_csv("progreso_terapia_combinado.csv", index=False)
        df_final, all_emociones = expandir_emociones(df_avance, emociones_por_sesion)
        df_final.to_csv("analisis_terapia_completo.csv", index=False)

        # Consolidate emociones DataFrame creation (avoid code duplication)
        import numpy as np
        pesos_df = pd.DataFrame(0, index=[e["n_sesion"] for e in emociones_por_sesion], columns=all_emociones)
        frases_df = pd.DataFrame("", index=[e["n_sesion"] for e in emociones_por_sesion], columns=all_emociones)
        for e in emociones_por_sesion:
            n = e["n_sesion"]
            for emo, data in e["emociones"].items():
                pesos_df.loc[n, emo] = data["peso"]
                frases_df.loc[n, emo] = data["frases"]

        resumenes = [
            resumen_chain.invoke({"objetivo": df_avance["objetivo_1"].iloc[0]}).content.strip(),
            resumen_chain.invoke({"objetivo": df_avance["objetivo_2"].iloc[0]}).content.strip(),
            resumen_chain.invoke({"objetivo": df_avance["objetivo_3"].iloc[0]}).content.strip()
        ]

        graficar_emociones(pesos_df, frases_df, all_emociones)
        graficar_progreso(df_avance, resumenes, filename=None)
        generar_pdf(df_avance, resumenes)

        st.sidebar.success("✅ Análisis ejecutado correctamente. Recargando datos...")
        st.rerun()
    except Exception as e:
        import traceback
        st.sidebar.error(f"❌ Error al ejecutar el análisis:\n{e}\n{traceback.format_exc()}")

# --- Barra lateral: Filtros y descarga ---
st.sidebar.header("Opciones")
modo = st.sidebar.radio(
    "¿Qué deseas ver?",
    [
        "Resumen general",
        "Detalle por sesión",
        "Chatbot clínico",
        "Árbol de conocimiento (YouTube)"
    ]
)
pdf_path = "informe_terapia_teo.pdf"
if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        st.sidebar.download_button(
            label="📄 Descargar informe PDF",
            data=f,
            file_name="informe_terapia_teo.pdf",
            mime="application/pdf"
        )
else:
    st.sidebar.warning("El informe PDF aún no está disponible.")

# --- Cargar datos principales justo antes de ser necesarios ---
if modo in ["Resumen general", "Detalle por sesión"]:
    df = pd.read_csv("analisis_terapia_completo.csv")
    if modo == "Detalle por sesión":
        sesiones = df["n_sesion"].tolist()
        sesion_seleccionada = st.sidebar.selectbox("Selecciona una sesión", sesiones)
        df_vista = df[df["n_sesion"] == sesion_seleccionada]
    else:
        df_vista = df

# --- Vista principal: Resumen general ---
if modo == "Resumen general":
    st.title("🧑‍⚕️ Informe de Progreso Terapéutico")
    st.markdown('<hr style="border-top: 2px solid #A3C9A8; margin-top: 0.5em; margin-bottom: 1em;">', unsafe_allow_html=True)
    st.markdown("### Hipótesis inicial del motivo de consulta")
    st.info(df["hipotesis_motivo_consulta"].iloc[0])
    st.markdown("### Objetivos terapéuticos")
    st.markdown(f"1. {df['objetivo_1'].iloc[0]}")
    st.markdown(f"2. {df['objetivo_2'].iloc[0]}")
    st.markdown(f"3. {df['objetivo_3'].iloc[0]}")
    st.markdown("### Cuadro de avances por sesión")
    df_porcentaje = avances_a_porcentaje(df, ["avance_objetivo_1", "avance_objetivo_2", "avance_objetivo_3"])
    st.dataframe(
        df_porcentaje[["n_sesion", "fecha", "avance_objetivo_1", "avance_objetivo_2", "avance_objetivo_3"]],
        use_container_width=True,
        hide_index=True
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Evolución del Progreso por Objetivo")
        resumenes = [
            resumen_chain.invoke({"objetivo": df["objetivo_1"].iloc[0]}).content.strip(),
            resumen_chain.invoke({"objetivo": df["objetivo_2"].iloc[0]}).content.strip(),
            resumen_chain.invoke({"objetivo": df["objetivo_3"].iloc[0]}).content.strip()
        ]
        graficar_progreso(df, resumenes, filename=None)
    with col2:
        st.markdown("#### Emociones por sesión")
        emocion_cols = [col for col in df.columns if col.startswith("peso_")]
        if emocion_cols:
            fig_emo = graficar_emociones_por_sesion(df, emocion_cols, EMOCION_COLORES, PASTEL_COLORS)
            st.plotly_chart(fig_emo, use_container_width=True)
        else:
            st.info("No hay datos de emociones para mostrar.")
    st.markdown("### Cierre de la última sesión")
    st.success(df['comentario_terapeutico'].iloc[-1])

# --- Vista de detalle por sesión ---
if modo == "Detalle por sesión":
    row = df_vista.iloc[0]
    st.title(f"📝 Detalle de la sesión {row['n_sesion']}")
    st.markdown(f"**Fecha:** {row['fecha']}")
    st.markdown(f"**Avance Objetivo 1:** {(row['avance_objetivo_1']*10):.1f} %")
    st.markdown(f"**Avance Objetivo 2:** {(row['avance_objetivo_2']*10):.1f} %")
    st.markdown(f"**Avance Objetivo 3:** {(row['avance_objetivo_3']*10):.1f} %")
    st.markdown("**Comentario terapéutico:**")
    st.info(row["comentario_terapeutico"])
    emocion_cols = [col for col in df.columns if col.startswith("peso_")]
    if emocion_cols:
        fig_emo, emociones, frases, pesos = graficar_emociones_detalle(row, emocion_cols, EMOCION_COLORES, PASTEL_COLORS)
        if fig_emo:
            st.plotly_chart(fig_emo, use_container_width=True)
        else:
            st.info("No se detectaron emociones para esta sesión.")
        if emociones:
            st.markdown("#### Frases principales de emociones:")
            for emo, frase, peso in zip(emociones, frases, pesos):
                st.markdown(f"**{emo.capitalize()} ({peso*100:.1f}%):** _{frase}_")

# --- Chatbot clínico ---
if modo == "Chatbot clínico":
    from chatbot import chatbot_interface_dashboard
    chatbot_interface_dashboard()

# --- Árbol de conocimiento (YouTube) ---
if modo == "Árbol de conocimiento (YouTube)":
    st.title("📚 Árbol de conocimiento (YouTube)")
    st.markdown('<hr style="border-top: 2px solid #A3C9A8; margin-top: 0.5em; margin-bottom: 1em;">', unsafe_allow_html=True)
    st.write("Explora el árbol de conocimiento relacionado con el caso clínico y los videos de terapia.")
    if st.button("🔄 Actualizar árbol de conocimiento"):
        with st.spinner("Actualizando árbol de conocimiento..."):
            result = subprocess.run(
                [os.sys.executable, "cargar_youtube.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("✅ Árbol de conocimiento actualizado correctamente. Recarga la página para ver los nuevos datos.")
            else:
                st.error(f"❌ Error al actualizar el árbol:\n{result.stderr}")
    mostrar_arbol_conocimiento_desde_csv("youtube_psicologia/videos_db.csv")

# --- Barra lateral: carga de videos y sesiones ---
with st.sidebar:
    st.header("Carga de videos de YouTube")
    video_url_input = st.text_area(
        "Pega aquí una o varias URLs de YouTube (una por línea) para cargar nuevas transcripciones:",
        height=100,
        placeholder="https://youtu.be/VIDEOID1\nhttps://youtu.be/VIDEOID2"
    )
    if st.button("Cargar videos"):
        urls = [url.strip() for url in video_url_input.splitlines() if url.strip()]
        if urls:
            result = subprocess.run(
                [os.sys.executable, "cargar_youtube.py"] + urls,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("✅ Videos cargados correctamente. Recarga la página para ver los nuevos datos.")
            else:
                st.error(f"❌ Error al cargar videos:\n{result.stderr}")
        else:
            st.warning("Por favor, ingresa al menos una URL de YouTube.")

    st.header("Carga de sesiones nuevas")
    st.markdown(
        """
        <small>
        <b>Nota importante:</b><br>
        • El <b>nombre del archivo</b> debe contener el número de sesión (ej: <b>Sesion_3.txt</b>).<br>
        • Es recomendable que el <b>texto del archivo</b> incluya la línea <b>Fecha: YYYY-MM-DD</b> o <b>Fecha: DD/MM/YYYY</b> para registrar la fecha de la sesión.<br>
        Si no se encuentra la fecha en el texto, se usará la fecha de creación del archivo.
        </small>
        """,
        unsafe_allow_html=True
    )
    carpeta_sesiones = st.text_input(
        "Nombre de la carpeta con archivos .txt de sesiones (debe ser igual al nombre del paciente, con la primera letra en mayúscula):",
        value="Teo"
    )
    paciente_id = st.text_input(
        "ID del paciente (para la colección):",
        value="teo"
    )
    st.session_state["paciente_id"] = paciente_id
    if st.button("Cargar sesiones nuevas"):
        archivos_txt = glob.glob(f"{carpeta_sesiones}/*.txt")
        if not archivos_txt:
            st.warning("No se encontraron archivos .txt en la carpeta indicada.")
        elif not paciente_id.strip():
            st.warning("Debes ingresar un ID de paciente.")
        else:
            result = subprocess.run(
                [
                    os.sys.executable, "Cargar_sesiones.py",
                    "--carpeta", carpeta_sesiones,
                    "--paciente", paciente_id
                ],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("✅ Sesiones cargadas correctamente. Recarga la página para ver los nuevos datos.")
            else:
                st.error(f"❌ Error al cargar sesiones:\n{result.stderr}")

    uploaded_files = st.file_uploader(
        "Arrastra aquí archivos .txt de sesión para subirlos a la carpeta seleccionada:",
        type="txt",
        accept_multiple_files=True
    )
    if uploaded_files:
        os.makedirs(carpeta_sesiones, exist_ok=True)
        for uploaded_file in uploaded_files:
            save_path = os.path.join(carpeta_sesiones, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} archivo(s) guardado(s) en '{carpeta_sesiones}'.")

    # Casillero para pegar la API Key de OpenAI
    openai_key = st.text_input(
        "🔑 Ingresa tu OpenAI API Key:",
        type="password",
        help="Tu clave nunca se guarda, solo se usa en esta sesión."
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
