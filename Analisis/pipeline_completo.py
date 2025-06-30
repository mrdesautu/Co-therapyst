import pandas as pd
from procesamiento import analizar_sesiones, expandir_emociones
from ejecutar_flujo_agentes import main as ejecutar_longitudinal

def ejecutar_procesamiento_y_guardar(sesiones, analisis_chain, emo_chain, retriever, embedding_model, client, llm):
    df_avance, emociones_por_sesion = analizar_sesiones(
        sesiones, analisis_chain, emo_chain, retriever, embedding_model, client, llm
    )
    df_final, _ = expandir_emociones(df_avance, emociones_por_sesion)
    df_final.to_csv("resultados_transversales.csv", index=False)
    return df_final

def ejecutar_longitudinal_y_guardar():
    # Modifica main() en ejecutar_flujo_agentes.py para que retorne un DataFrame o guarde un CSV
    ejecutar_longitudinal()  # Debe guardar 'resultados_longitudinales.csv'

def combinar_csvs():
    df_trans = pd.read_csv("resultados_transversales.csv")
    df_long = pd.read_csv("resultados_longitudinales.csv")
    # Realiza el merge por las columnas clave
    df_final = pd.merge(df_trans, df_long, on=["paciente", "n_sesion", "fecha"], how="outer")
    df_final.to_csv("resultados_combinados.csv", index=False)
    print("CSV combinado guardado como resultados_combinados.csv")

if __name__ == "__main__":
    # Aquí debes preparar tus objetos y sesiones como en los scripts originales
    # sesiones = ...
    # analisis_chain, emo_chain, retriever, embedding_model, client, llm = ...
    # ejecutar_procesamiento_y_guardar(sesiones, analisis_chain, emo_chain, retriever, embedding_model, client, llm)
    ejecutar_longitudinal_y_guardar()
    combinar_csvs()