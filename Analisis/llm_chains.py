import os
import csv
import logging
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# --- Configuración y modelos ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# --- Prompts base ---
PROMPT_ANALISIS_BASE = """
{instruccion}

Por favor responde SIEMPRE en el siguiente formato:

1. Hipótesis diagnóstica:
<texto de hipótesis>

2. Propuesta de tratamiento:
<texto de tratamiento>

3. Informe de avances:
<texto de análisis general>

Contexto relevante:
{context}

Sesión:
{session}

Pregunta:
{question}

Respuesta:
"""

PROMPT_INTEGRADOR_BASE = """
{instruccion}

Has recibido las siguientes visiones generales del caso, cada una desde una perspectiva teórica distinta:

{visiones}

Por favor responde SIEMPRE en el siguiente formato:

1. Hipótesis diagnóstica:
<texto de hipótesis>

2. Propuesta de tratamiento:
<texto de tratamiento>

3. Informe de avances:
<texto de análisis general>

Contexto adicional:
{context}

Sesión:
{session}

Visión integrada:
"""

PROMPT_PREGUNTAS_SUGERIDAS = """
{instruccion}

Basado en el análisis de la sesión, sugiere 3 preguntas potentes o estratégicas que el terapeuta podría plantear en la próxima sesión para profundizar el proceso terapéutico. 
Responde solo con una lista numerada de 3 preguntas, sin explicaciones.

Contexto relevante:
{context}

Sesión:
{session}

Preguntas sugeridas:
"""

PROMPT_FINAL_INTEGRADOR = """
Eres un supervisor clínico experto. Has recibido las siguientes visiones generales del caso, cada una desde una perspectiva teórica distinta:

{visiones}

Por favor, sintetiza los aspectos centrales destacados en las visiones anteriores y genera una última visión global integradora, resaltando los puntos de convergencia y los aportes únicos de cada enfoque.

Responde en un solo bloque de texto claro y estructurado.
"""

# --- Perspectivas teóricas ---
PERSPECTIVAS = [
    (
        "Cognitivo-Conductual",
        "Eres un psicólogo clínico especializado en terapia cognitivo-conductual (TCC). "
        "Analiza la sesión identificando pensamientos automáticos, creencias disfuncionales, emociones asociadas y conductas problema. "
        "Formula hipótesis clínicas, objetivos terapéuticos y una propuesta de tratamiento desde la TCC."
    ),
    (
        "Sistémico",
        "Eres un terapeuta sistémico. Analiza la sesión observando patrones de interacción, roles familiares, jerarquías, ciclos vitales y lealtades vinculares. "
        "Enfócate en las dinámicas relacionales más que en el individuo aislado. Formula hipótesis sistémicas y posibles intervenciones contextuales."
    ),
    (
        "Psicología Positiva",
        "Eres un terapeuta con orientación en psicología positiva. Identifica en la sesión fortalezas personales, recursos internos y sociales del paciente, "
        "así como emociones positivas, sentido de vida y potencialidades no desarrolladas. Propón habilidades a entrenar y objetivos orientados al florecimiento humano."
    ),
    (
        "Lacaniano",
        "Eres un psicoanalista lacaniano (sin utilizar la teoría de Freud ni de Miller). Interpreta el discurso del sujeto a través de los registros Real, Simbólico e Imaginario, "
        "la cadena significante, los cortes en el habla y los lapsus. Considera el lugar del sujeto en el Otro y la función de la demanda. "
        "No traduzcas ni simplifiques: produce una hipótesis en lenguaje clínico lacaniano."
    ),
    (
        "Existencial",
        "Eres un terapeuta existencial con base en la filosofía de Martin Heidegger. Reflexiona sobre la sesión considerando nociones como el ser-en-el-mundo, la autenticidad, la angustia existencial, "
        "el proyecto vital, la muerte y el Dasein. Tu análisis debe centrarse en el sentido, la libertad y la responsabilidad del paciente ante su existencia."
    )
]

# --- Factories para agentes ---
def crear_teorico_agents():
    return [
        TeoricoAgent(nombre, PROMPT_ANALISIS_BASE, instruccion)
        for nombre, instruccion in PERSPECTIVAS
    ]

def crear_integrador_agents():
    return [
        IntegradorVisionAgent(nombre, PROMPT_INTEGRADOR_BASE, instruccion)
        for nombre, instruccion in PERSPECTIVAS
    ]

def crear_preguntas_sugeridas_agents():
    return [
        PreguntasSugeridasAgent(nombre, PROMPT_PREGUNTAS_SUGERIDAS, instruccion)
        for nombre, instruccion in PERSPECTIVAS
    ]

def crear_vision_general_agents():
    return [
        VisionGeneralAgent(nombre, f"""
Eres un experto en {nombre}.
Genera una visión general del caso desde tu perspectiva teórica.
{{context}}

Sesión:
{{session}}

Visión general:
""")
        for nombre, _ in PERSPECTIVAS
    ]

# --- Clases de agentes ---
class TeoricoAgent:
    """
    Agente teórico que realiza análisis clínico desde una perspectiva específica.
    """
    def __init__(self, perspectiva: str, prompt_template: str, instruccion: str = None):
        self.perspectiva = perspectiva
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | llm
        self.instruccion = instruccion

    def analizar_sesion(self, context: str, session: str, question: str) -> str:
        return self.chain.invoke({
            "instruccion": self.instruccion,
            "context": context,
            "session": session,
            "question": question
        })

class IntegradorVisionAgent(TeoricoAgent):
    """
    Agente que integra visiones generales de todos los enfoques, generando una síntesis desde su perspectiva.
    """
    def __init__(self, perspectiva: str, prompt_template: str, instruccion: str = None):
        super().__init__(perspectiva, prompt_template, instruccion)

    def integrar_visiones(self, visiones: str, context: str, session: str) -> str:
        return self.chain.invoke({
            "instruccion": self.instruccion,
            "context": context,
            "session": session,
            "visiones": visiones
        })

class PreguntasSugeridasAgent:
    """
    Agente que sugiere 3 preguntas para la próxima sesión desde una perspectiva teórica.
    """
    def __init__(self, perspectiva: str, prompt_template: str, instruccion: str = None):
        self.perspectiva = perspectiva
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | llm
        self.instruccion = instruccion

    def sugerir_preguntas(self, context: str, session: str) -> str:
        return self.chain.invoke({
            "instruccion": self.instruccion,
            "context": context,
            "session": session
        })

class VisionGeneralAgent(TeoricoAgent):
    """
    Agente que genera una visión general del caso desde una perspectiva teórica.
    """
    def vision_general(self, context: str, session: str) -> str:
        return self.chain.invoke({
            "context": context,
            "session": session,
            "question": "Genera una visión general del caso desde tu perspectiva teórica."
        })

class AgenteIntegradorFinal:
    """
    Agente que integra las 5 visiones generales en una última visión global.
    """
    def __init__(self, llm):
        self.prompt = PromptTemplate.from_template(PROMPT_FINAL_INTEGRADOR)
        self.chain = self.prompt | llm

    def integrar(self, visiones_dict):
        visiones_txt = "\n".join([f"{k}: {v}" for k, v in visiones_dict.items()])
        return self.chain.invoke({"visiones": visiones_txt})

# --- Chains auxiliares ---
prompt = PromptTemplate.from_template("""
Eres un psicólogo clínico desde la psicología positiva con muchos años de experiencia.
Tienes acceso a contexto relevante en tu base de conocimiento.

Información de apoyo:
{context}

Texto de la sesión:
{session}

Pregunta:
{question}

Respuesta:
""")
analisis_chain = prompt | llm

prompt_emo = PromptTemplate.from_template("""
Analiza las emociones predominantes que expresa el paciente en el siguiente texto de sesión,
considerando también la información de apoyo.

Contexto:
{context}

Texto sesión:
{session}

Para cada emoción, asigna un peso entre 0 y 1 y proporciona una o dos frases representativas del paciente
que ejemplifiquen esa emoción, en el formato:

emoción1: 0.5, frases: "frase 1, frase 2"
emoción2: 0.3, frases: "frase 1, frase 2"
emoción3: 0.2, frases: "frase 1, frase 2"

Emociones con peso y frases:
""")
emo_chain = prompt_emo | llm

prompt_resumen = PromptTemplate.from_template("""
Eres un terapeuta experto en psicología positiva. Resume el siguiente objetivo terapéutico en un título de máximo 5 palabras, manteniendo su esencia.

Objetivo: {objetivo}

Resumen:
""")
resumen_chain = prompt_resumen | llm

# --- Funciones principales ---
def generar_analisis_completo_y_revisado(context: str, session: str, question: str, max_intentos=5, csv_path="analisis_sesiones.csv") -> dict:
    """
    Ejecuta el análisis y revisión para cada perspectiva teórica, guarda la trazabilidad en CSV.
    """
    agentes = crear_teorico_agents()
    revisores = [RevisorAgent(nombre) for nombre, _ in PERSPECTIVAS]
    resultados = {}
    trazabilidad = []

    for agente, revisor in zip(agentes, revisores):
        intentos = 0
        while True:
            intentos += 1
            analisis = agente.analizar_sesion(context, session, question)
            puntaje, justificacion = revisor.puntuar(analisis)
            hipotesis, tratamiento, analisis_general = extraer_partes_analisis(analisis)
            paciente, n_sesion, fecha = extraer_metadata_context(context)
            trazabilidad.append({
                "paciente": paciente,
                "n_sesion": n_sesion,
                "fecha": fecha,
                "perspectiva": agente.perspectiva,
                "diagnostico": hipotesis,
                "hipotesis": hipotesis,
                "tratamiento": tratamiento,
                "analisis_general": analisis_general,
                "puntaje": puntaje,
                "justificacion": justificacion,
                "intentos": intentos,
                "question": question
            })
            if puntaje >= 8 or intentos >= max_intentos:
                break
        resultados[agente.perspectiva] = {
            "analisis": analisis,
            "puntaje": puntaje,
            "justificacion": justificacion,
            "intentos": intentos
        }

    # Guardar en CSV
    fieldnames = [
        "paciente", "n_sesion", "fecha", "perspectiva",
        "diagnostico", "hipotesis", "tratamiento", "analisis_general",
        "puntaje", "justificacion", "intentos", "question"
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in trazabilidad:
            writer.writerow(row)

    return resultados

def generar_visiones_generales(context: str, session: str) -> dict:
    """
    Genera visiones generales del caso desde todas las perspectivas teóricas.
    """
    agentes = crear_vision_general_agents()
    resultados = {}
    for agente in agentes:
        try:
            output = agente.vision_general(context, session)
            resultados[agente.perspectiva] = output
        except Exception as e:
            resultados[agente.perspectiva] = f"Error en visión general: {str(e)}"
    return resultados

def generar_visiones_integradas(context: str, session: str, visiones_generales: dict) -> dict:
    """
    Cada agente integrador recibe todas las visiones generales y genera una síntesis desde su perspectiva.
    """
    visiones_str = "\n".join([f"{k}: {v}" for k, v in visiones_generales.items()])
    agentes = crear_integrador_agents()
    resultados = {}
    for agente in agentes:
        try:
            output = agente.integrar_visiones(visiones_str, context, session)
            resultados[agente.perspectiva] = output
        except Exception as e:
            resultados[agente.perspectiva] = f"Error en visión integrada: {str(e)}"
    return resultados

def generar_vision_global_final(visiones_generales: dict) -> str:
    """
    Genera una visión global final integrando las 5 visiones generales.
    """
    agente_final = AgenteIntegradorFinal(llm)
    return agente_final.integrar(visiones_generales)

def generar_preguntas_sugeridas(context: str, session: str) -> dict:
    """
    Genera 3 preguntas sugeridas para la próxima sesión desde todas las perspectivas teóricas.
    """
    agentes = crear_preguntas_sugeridas_agents()
    resultados = {}
    for agente in agentes:
        try:
            output = agente.sugerir_preguntas(context, session)
            resultados[agente.perspectiva] = output
        except Exception as e:
            resultados[agente.perspectiva] = f"Error generando preguntas: {str(e)}"
    return resultados

# --- Agentes revisores ---
class RevisorAgent:
    """
    Agente que revisa un análisis y asigna puntaje y justificación.
    """
    def __init__(self, perspectiva: str):
        self.perspectiva = perspectiva
        self.prompt = PromptTemplate.from_template(f"""
Eres un revisor experto en la perspectiva {perspectiva}.
Lee el siguiente análisis y asígnale un puntaje del 1 al 10 según su calidad, profundidad y relevancia clínica.
Justifica brevemente tu puntaje.

Análisis:
{{analisis}}

Responde SOLO en el formato:
puntaje: <número>
justificación: <texto>
""")
        self.chain = self.prompt | llm

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

# --- Utilidades de parseo y metadatos ---
def extraer_partes_analisis(analisis: str):
    """
    Extrae hipótesis, tratamiento y análisis general del output del LLM.
    """
    if hasattr(analisis, "content"):
        analisis = analisis.content
    regex = (
        r"Hip[oó]tesis diagn[oó]stica:\s*(.*?)\n\s*2\.\s*Propuesta de tratamiento:\s*(.*?)\n\s*3\.\s*(Informe de avances|Informe de avance|An[aá]lisis general):\s*(.*)"
    )
    match = re.search(regex, analisis, re.DOTALL | re.IGNORECASE)
    if match:
        hipotesis = match.group(1).strip()
        tratamiento = match.group(2).strip()
        analisis_general = match.group(4).strip()
    else:
        hipotesis = ""
        tratamiento = ""
        analisis_general = analisis.strip()
    return hipotesis, tratamiento, analisis_general

def extraer_metadata_context(context):
    """
    Extrae paciente, número de sesión y fecha del contexto.
    Ejemplo de contexto: "Paciente: teo. Sesión número: 1. Fecha: 2025-03-22."
    """
    paciente, n_sesion, fecha = "", "", ""
    m = re.search(r"Paciente:\s*([^\.\n]+)", context)
    if m: paciente = m.group(1).strip()
    m = re.search(r"Sesión número:\s*([^\.\n]+)", context)
    if m: n_sesion = m.group(1).strip()
    m = re.search(r"Fecha:\s*([^\.\n]+)", context)
    if m: fecha = m.group(1).strip()
    return paciente, n_sesion, fecha