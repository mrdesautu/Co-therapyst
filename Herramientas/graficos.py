import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import make_interp_spline
import logging
import streamlit as st

logger = logging.getLogger(__name__)

def graficar_emociones(pesos_df, frases_df, all_emociones, filename="grafico_emociones.png"):
    fig = go.Figure()
    sesiones_idx = sorted(pesos_df.index)
    color_map = {emo: px.colors.qualitative.Plotly[i % 10] for i, emo in enumerate(all_emociones)}
    for emo in all_emociones:
        y = pesos_df[emo].reindex(sesiones_idx).fillna(0).values
        hover_texts = frases_df[emo].reindex(sesiones_idx).fillna("").values
        text_labels = [emo if val > 0.05 else "" for val in y]
        fig.add_trace(go.Bar(
            x=sesiones_idx,
            y=y,
            name=emo,
            text=text_labels,
            hovertext=hover_texts,
            hoverinfo='text+y',
            textposition='inside',
            marker_color=color_map[emo]
        ))
    fig.update_layout(
        barmode='stack',
        title="Emociones por sesión con fragmentos del paciente",
        xaxis_title="Número de sesión",
        yaxis_title="Proporción de emociones"
    )
    fig.write_image(filename, width=900, height=500)
    fig.show()

def graficar_progreso(df_avance, resumenes, filename=None):
    plt.figure(figsize=(10, 6))
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    marcadores = ['o', 's', '^']
    estilos = ['-', '--', '-.']
    offset = [0, 0.08, -0.08]
    sesiones = df_avance["n_sesion"].values
    for i, col in enumerate(["avance_objetivo_1", "avance_objetivo_2", "avance_objetivo_3"]):
        y = df_avance[col].values.astype(float) + offset[i]
        if len(sesiones) > 3:
            xnew = np.linspace(sesiones.min(), sesiones.max(), 200)
            spline = make_interp_spline(sesiones, y, k=3)
            y_smooth = spline(xnew)
            plt.plot(xnew, y_smooth, label=resumenes[i], color=colores[i], linewidth=2, linestyle=estilos[i])
            plt.scatter(sesiones, y, color=colores[i], marker=marcadores[i], s=70)
        else:
            plt.plot(sesiones, y, label=resumenes[i], marker=marcadores[i], color=colores[i])
    plt.title("Evolución del Progreso Terapéutico")
    plt.xlabel("Sesión")
    plt.ylabel("Progreso (0 a 10)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        st.pyplot(plt)
    plt.close()