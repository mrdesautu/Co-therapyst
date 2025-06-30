import os
import re
from llm_chains import (
    generar_analisis_completo_y_revisado,
    generar_visiones_generales,
    generar_visiones_integradas,
    llm
)
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

def leer_sesiones_chromadb(paciente_id="teo"):
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
        sesiones.append({
            "doc_id": doc_id,
            "texto": texto,
            "fecha": meta.get("fecha", ""),
            "n_sesion": meta.get("n_sesion", doc_id)
        })
    sesiones.sort(key=lambda x: int(x["n_sesion"]) if str(x["n_sesion"]).isdigit() else x["n_sesion"])
    return sesiones

# --- Agentes longitudinales ---

class LongitudinalAgent:
    def __init__(self, perspectiva, prompt_template_str, llm):
        self.perspectiva = perspectiva
        self.prompt = PromptTemplate.from_template(prompt_template_str)
        self.chain = self.prompt | llm

    def evaluar_vigencia(self, analisis_previos, session, context):
        entrada = {
            "analisis_previos": analisis_previos,
            "session": session,
            "context": context
        }
        respuesta = self.chain.invoke(entrada)
        if hasattr(respuesta, "content"):
            respuesta = respuesta.content
        return respuesta

    def puntuar(self, analisis: str) -> tuple:
        respuesta = self.chain.invoke({"analisis": analisis})
        if hasattr(respuesta, "content"):
            respuesta = respuesta.content
        match = re.search(r'puntaje:\s*(\d+).*justificación:\s*(.*)', respuesta, re.DOTALL | re.IGNORECASE)
        if match:
            puntaje = int(match.group(1))
            justificacion = match.group(2).strip()
        else:
            puntaje = 5
            justificacion = "No se pudo extraer justificación."
        return puntaje, justificacion

PROMPT_LONGITUDINAL = {
    "Cognitivo-Conductual": """
Eres un psicólogo cognitivo-conductual. Evalúa la vigencia de los análisis previos en base a la nueva sesión.
Analisis previos:
{analisis_previos}

Nueva sesión:
{session}

Contexto:
{context}

Responde SOLO en el formato:
puntaje: <número>
justificación: <texto>
nuevo_motivo: <texto si puntaje < 8, si no deja vacío>
""",
    "Sistémico": """
Eres un terapeuta sistémico. Evalúa la vigencia de los análisis previos en base a la nueva sesión.
Analisis previos:
{analisis_previos}

Nueva sesión:
{session}

Contexto:
{context}

Responde SOLO en el formato:
puntaje: <número>
justificación: <texto>
nuevo_motivo: <texto si puntaje < 8, si no deja vacío>
""",
    "Psicología Positiva": """
Eres un psicólogo positivo. Evalúa la vigencia de los análisis previos en base a la nueva sesión.
Analisis previos:
{analisis_previos}

Nueva sesión:
{session}

Contexto:
{context}

Responde SOLO en el formato:
puntaje: <número>
justificación: <texto>
nuevo_motivo: <texto si puntaje < 8, si no deja vacío>
""",
    "Lacaniano": """
Eres un analista lacaniano. Evalúa la vigencia de los análisis previos en base a la nueva sesión.
Analisis previos:
{analisis_previos}

Nueva sesión:
{session}

Contexto:
{context}

Responde SOLO en el formato:
puntaje: <número>
justificación: <texto>
nuevo_motivo: <texto si puntaje < 8, si no deja vacío>
""",
    "Existencial": """
Eres un terapeuta existencial. Evalúa la vigencia de los análisis previos en base a la nueva sesión.
Analisis previos:
{analisis_previos}

Nueva sesión:
{session}

Contexto:
{context}

Responde SOLO en el formato:
puntaje: <número>
justificación: <texto>
nuevo_motivo: <texto si puntaje < 8, si no deja vacío>
"""
}

def evaluar_longitudinal(sesiones, resultados_por_sesion, llm):
    longitudinales = {
        nombre: LongitudinalAgent(nombre, PROMPT_LONGITUDINAL[nombre], llm)
        for nombre in PROMPT_LONGITUDINAL
    }
    resultados_longitudinales = []
    for i in range(1, len(sesiones)):
        context = f"Paciente: {sesiones[i]['doc_id']}. Sesión número: {sesiones[i]['n_sesion']}. Fecha: {sesiones[i]['fecha']}."
        session_text = sesiones[i]["texto"]
        prev_analisis = resultados_por_sesion[i-1]
        resultado_sesion = {}
        for perspectiva, agente in longitudinales.items():
            analisis_previos = prev_analisis.get(perspectiva, {}).get("analisis", "")
            salida = agente.evaluar_vigencia(analisis_previos, session_text, context)
            if hasattr(salida, "content"):
                salida = salida.content
            match = re.search(r'puntaje:\s*(\d+).*justificación:\s*(.*)nuevo_motivo:\s*(.*)', salida, re.DOTALL | re.IGNORECASE)
            if match:
                puntaje = int(match.group(1))
                justificacion = match.group(2).strip()
                nuevo_motivo = match.group(3).strip()
            else:
                puntaje = 5
                justificacion = "No se pudo extraer justificación."
                nuevo_motivo = ""
            resultado_sesion[perspectiva] = {
                "puntaje": puntaje,
                "justificacion": justificacion,
                "nuevo_motivo": nuevo_motivo
            }
        resultados_longitudinales.append(resultado_sesion)
    return resultados_longitudinales

def main():
    paciente_id = "teo"
    sesiones = leer_sesiones_chromadb(paciente_id)
    print(f"Se encontraron {len(sesiones)} sesiones para el paciente '{paciente_id}'.")

    resultados_por_sesion = []
    for sesion in sesiones:
        context = f"Paciente: {paciente_id}. Sesión número: {sesion['n_sesion']}. Fecha: {sesion['fecha']}."
        session_text = sesion["texto"]
        question = "¿Cuál es el análisis clínico de esta sesión?"

        print(f"\n=== Analizando sesión {sesion['n_sesion']} ({sesion['fecha']}) ===")
        resultados_analisis = generar_analisis_completo_y_revisado(context, session_text, question)
        resultados_por_sesion.append(resultados_analisis)
        for perspectiva, datos in resultados_analisis.items():
            print(f"\n[{perspectiva}]")
            print(f"Análisis: {datos['analisis']}")
            print(f"Puntaje: {datos['puntaje']} ({datos['justificacion']})")
            print(f"Intentos: {datos['intentos']}")

        print("\n=== Visiones generales de cada teoría ===")
        visiones = generar_visiones_generales(context, session_text)
        for perspectiva, vision in visiones.items():
            print(f"\n[{perspectiva}]\n{vision}")

        print("\n=== Visiones integradas de cada teoría ===")
        visiones_integradas = generar_visiones_integradas(context, session_text, visiones)
        for perspectiva, vision in visiones_integradas.items():
            print(f"\n[{perspectiva}]\n{vision}")

        print("\n--- Fin de análisis de la sesión ---\n")

    # FASE 3: Evaluación longitudinal
    print("\n=== FASE 3: Evaluación longitudinal ===")
    resultados_longitudinales = evaluar_longitudinal(sesiones, resultados_por_sesion, llm)
    for i, resultado in enumerate(resultados_longitudinales, start=2):
        print(f"\nEvaluación longitudinal para sesión {i}:")
        for perspectiva, datos in resultado.items():
            print(f"[{perspectiva}] Puntaje: {datos['puntaje']} | Justificación: {datos['justificacion']}")
            if datos['puntaje'] < 8:
                print(f"Nuevo motivo propuesto: {datos['nuevo_motivo']}")

if __name__ == "__main__":
    main()