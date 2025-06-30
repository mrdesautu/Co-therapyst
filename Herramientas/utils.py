import os
import random
import plotly.graph_objects as go

def get_emocion_color(emo, EMOCION_COLORES, PASTEL_COLORS):
    """Devuelve el color asociado a una emoción, o uno pastel aleatorio si no está definida."""
    return EMOCION_COLORES.get(emo.lower(), random.choice(PASTEL_COLORS))

def avances_a_porcentaje(df, cols):
    """Convierte columnas de avance a porcentaje (escala 0-10 a 0-100%)."""
    df_copy = df.copy()
    for col in cols:
        df_copy[col] = (df_copy[col] * 10).round(1).astype(str) + " %"
    return df_copy

def graficar_emociones_por_sesion(df, emocion_cols, EMOCION_COLORES, PASTEL_COLORS):
    """Devuelve una figura Plotly con las emociones por sesión."""
    emociones = [col.replace("peso_", "") for col in emocion_cols]
    pesos = df[emocion_cols].fillna(0)
    fig_emo = go.Figure()
    for emo in emociones:
        color = get_emocion_color(emo, EMOCION_COLORES, PASTEL_COLORS)
        fig_emo.add_trace(go.Bar(
            x=df["n_sesion"],
            y=(pesos[f"peso_{emo}"]*100).round(1),
            name=emo,
            text=[emo.capitalize() if v > 0.05 else "" for v in pesos[f"peso_{emo}"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#6B9080", size=12),
            marker_color=color,
            hovertext=df[f"frases_{emo}"] if f"frases_{emo}" in df else None,
            hovertemplate='<b>%{x}</b><br>Peso: %{y:.1f} %<br>Frase: %{hovertext}<extra></extra>'
        ))
    fig_emo.update_layout(
        barmode='stack',
        xaxis_title="Sesión",
        yaxis_title="Proporción de emociones (%)",
        title="Emociones por sesión (hover para ver frases)",
        legend_title="Emociones",
        plot_bgcolor="#f7f6f2",
        paper_bgcolor="#f7f6f2",
        font=dict(color="#6B9080")
    )
    return fig_emo

def graficar_emociones_detalle(row, emocion_cols, EMOCION_COLORES, PASTEL_COLORS):
    """Devuelve una figura Plotly para el detalle de una sesión."""
    emociones, frases, pesos, colores = [], [], [], []
    for col in emocion_cols:
        emo = col.replace("peso_", "")
        peso = row[col]
        frase_col = f"frases_{emo}"
        frase = row[frase_col] if frase_col in row else ""
        if peso > 0:
            emociones.append(emo)
            frases.append(frase)
            pesos.append(peso)
            colores.append(get_emocion_color(emo, EMOCION_COLORES, PASTEL_COLORS))
    if emociones:
        fig_emo = go.Figure(data=[
            go.Bar(
                x=emociones,
                y=[p*100 for p in pesos],
                text=[emo.capitalize() for emo in emociones],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="#6B9080", size=14),
                cliponaxis=False,
                hovertemplate='<b>%{x}</b><br>Peso: %{y:.1f} %<br>Frase: %{text}<extra></extra>',
                marker_color=colores
            )
        ])
        fig_emo.update_layout(
            title="Emociones predominantes en la sesión (hover para ver frase)",
            xaxis_title="Emoción",
            yaxis_title="Proporción (%)",
            yaxis_range=[0, 100],
            plot_bgcolor="#f7f6f2",
            paper_bgcolor="#f7f6f2",
            font=dict(color="#6B9080")
        )
        return fig_emo, emociones, frases, pesos
    return None, [], [], []

def graficar_progreso(df_avance, resumenes, filename=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import make_interp_spline
    import matplotlib.ticker as ticker

    plt.figure(figsize=(10, 6))
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    marcadores = ['o', 's', '^']
    estilos = ['-', '--', '-.']
    offset = [0, 0.08, -0.08]
    sesiones = df_avance["n_sesion"].values

    for i, col in enumerate(["avance_objetivo_1", "avance_objetivo_2", "avance_objetivo_3"]):
        y = df_avance[col].values.astype(float) * 10 + offset[i]
        if len(sesiones) > 3:
            xnew = np.linspace(sesiones.min(), sesiones.max(), 200)
            spline = make_interp_spline(sesiones, y, k=3)
            y_smooth = spline(xnew)
            plt.plot(xnew, y_smooth, label=resumenes[i], color=colores[i], linewidth=2, linestyle=estilos[i])
            plt.scatter(sesiones, y, color=colores[i], marker=marcadores[i], s=70)
        else:
            plt.plot(sesiones, y, marker=marcadores[i], label=resumenes[i], color=colores[i], linewidth=2, linestyle=estilos[i], markersize=9)
    plt.title("Evolución del Progreso Terapéutico", fontsize=16, fontweight='bold')
    plt.xlabel("Sesión", fontsize=12)
    plt.ylabel("Progreso (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.xticks(df_avance["n_sesion"])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
        
EMOCION_COLORES = {
    "alegria": "#B5EAD7",
    "gratitud": "#FFFACD",
    "calma": "#A3C9A8",
    "orgullo": "#C7CEEA",
    "esperanza": "#FFD6BA",
    "amor": "#FFDAC1",
    "entusiasmo": "#B5EAD7",
    "confianza": "#A3C9A8",
    "curiosidad": "#C7CEEA",
    "satisfaccion": "#FFD6BA",
    "tristeza": "#FFB7B2",
    "miedo": "#B2C2BF",
    "ira": "#FF6961",
    "ansiedad": "#FF6961",
    "culpa": "#C23B22",
    "vergüenza": "#B39EB5",
    "soledad": "#A9A9A9",
    "frustracion": "#FFB347",
    "enojo": "#FF6961",
    "aburrimiento": "#B2C2BF",
    "desesperanza": "#B39EB5",
    "apatia": "#A9A9A9",
    "resentimiento": "#C23B22",
    "preocupacion": "#FFB347",
    "desanimo": "#B39EB5",
    "inseguridad": "#B2C2BF",
    "celos": "#FFB7B2",
    "odio": "#C23B22",
    "desconfianza": "#B2C2BF",
    "paz": "#A3C9A8",
    "motivacion": "#FFD6BA",
    "placer": "#FFDAC1",
    "compasion": "#B5EAD7",
    "orgullo": "#C7CEEA",
    "sorpresa": "#C7CEEA",
    "alivio": "#B5EAD7",
    "culpabilidad": "#C23B22",
    "hostilidad": "#FF6961",
    "temor": "#B2C2BF",
    "nostalgia": "#B39EB5",
    "afecto": "#FFD6BA",
    "solidaridad": "#B5EAD7",
    "euforia": "#FFD6BA",
    "placer": "#FFDAC1",
    "compromiso": "#A3C9A8",
    "optimismo": "#FFD6BA",
    "desilusion": "#B39EB5",
    "desprecio": "#C23B22",
    "culpable": "#C23B22",
    "hostil": "#FF6961",
    "temeroso": "#B2C2BF",
    "nostalgico": "#B39EB5",
    # ...agrega más si tu modelo detecta otras...
}

# --- Paleta de colores pastel ---
PASTEL_COLORS = [
    "#A3C9A8", "#FFD6BA", "#B5EAD7", "#FFDAC1", "#C7CEEA", "#FFB7B2", "#B5EAD7", "#E2F0CB", "#FFB347", "#B2C2BF"
]
