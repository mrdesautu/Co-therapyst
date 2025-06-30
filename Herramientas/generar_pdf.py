from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import logging

logger = logging.getLogger(__name__)

def generar_pdf(df_avance, resumenes, filename="informe_terapia.pdf"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename, pagesize=A4)
    contenido = [
        Paragraph("Informe de Progreso Terapéutico - Paciente", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"<b>Hipótesis:</b> {df_avance['hipotesis_motivo_consulta'].iloc[0]}", styles["BodyText"]),
        Spacer(1, 6),
        Paragraph("<b>Objetivos:</b>", styles["BodyText"]),
        Paragraph(f"1. {df_avance['objetivo_1'].iloc[0]}", styles["BodyText"]),
        Paragraph(f"2. {df_avance['objetivo_2'].iloc[0]}", styles["BodyText"]),
        Paragraph(f"3. {df_avance['objetivo_3'].iloc[0]}", styles["BodyText"]),
        Spacer(1, 12)
    ]
    tabla_datos = [["Sesión", "Fecha", "Obj. 1", "Obj. 2", "Obj. 3"]]
    for _, row in df_avance.iterrows():
        tabla_datos.append([
            int(row["n_sesion"]), row["fecha"],
            f'{row["avance_objetivo_1"]:.2f}',
            f'{row["avance_objetivo_2"]:.2f}',
            f'{row["avance_objetivo_3"]:.2f}'
        ])
    tabla = Table(tabla_datos)
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    contenido.extend([
        tabla,
        Spacer(1, 12),
        Paragraph("<b>Gráficos:</b>", styles["BodyText"]),
        Image("grafico_progreso_mejorado.png", width=400, height=250),
        Image("grafico_emociones.png", width=400, height=250),
        Spacer(1, 12),
        Paragraph("<b>Comentario último:</b>", styles["BodyText"]),
        Paragraph(df_avance['comentario_terapeutico'].iloc[-1], styles["BodyText"])
    ])
    doc.build(contenido)
    logger.info(f"PDF generado: {filename}")