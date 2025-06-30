import os
import subprocess
from dotenv import load_dotenv

import whisper
import librosa
import soundfile as sf
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.hparams import sampling_rate

load_dotenv()

def descargar_audio_youtube(url, carpeta_destino):
    os.makedirs(carpeta_destino, exist_ok=True)
    salida = os.path.join(carpeta_destino, "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "mp3",
        "-o", salida,
        url
    ]
    print("Descargando audio de YouTube...")
    subprocess.run(cmd, check=True)
    archivos = [f for f in os.listdir(carpeta_destino) if f.endswith(".mp3")]
    if not archivos:
        raise FileNotFoundError("No se encontró el archivo de audio descargado.")
    return os.path.join(carpeta_destino, archivos[0])

def transcribir_audio(audio_path, modelo="base"):
    model = whisper.load_model(modelo)
    print(f"Transcribiendo: {audio_path} ...")
    result = model.transcribe(audio_path, language="es")
    return result["text"]

def guardar_texto(texto, carpeta_destino, nombre_base):
    os.makedirs(carpeta_destino, exist_ok=True)
    nombre_archivo = os.path.join(carpeta_destino, f"{nombre_base}.txt")
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"✅ Transcripción guardada en: {nombre_archivo}")

def diarizar_audio(audio_path, num_speakers=2, max_duration=60):
    wav, sr = librosa.load(audio_path, sr=sampling_rate, duration=max_duration)
    sf.write("temp.wav", wav, sampling_rate)
    wav = preprocess_wav("temp.wav")
    encoder = VoiceEncoder()
    embeds, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(cont_embeds)
    labels = clustering.labels_

    segment_duration = len(wav) / sampling_rate / len(labels)
    segments = []
    for i, label in enumerate(labels):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        segments.append({
            "start": start,
            "end": end,
            "speaker": f"SPEAKER_{label:02d}"
        })
    os.remove("temp.wav")
    # Filtra segmentos muy cortos
    MIN_SEG_DUR = 2.0  # segundos
    segments = [s for s in segments if (s["end"] - s["start"]) >= MIN_SEG_DUR]
    return segments

def transcribir_segmentos(audio_path, segments, modelo="tiny"):
    model = whisper.load_model(modelo)
    results = []
    for seg in segments:
        seg_file = f"temp_{seg['start']:.2f}_{seg['end']:.2f}.wav"
        os.system(f"ffmpeg -y -i \"{audio_path}\" -ss {seg['start']} -to {seg['end']} -ar 16000 -ac 1 -vn {seg_file}")
        result = model.transcribe(seg_file, language="es")
        results.append((seg['speaker'], result["text"]))
        os.remove(seg_file)
    return results

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=YmnZISnZh84&ab_channel=PsycastES"
    carpeta_audio = "sesiones_de_videos"
    carpeta_texto = "sesiones_de_videos"
    num_speakers = 2  # Ajusta según el video

    audio_path = descargar_audio_youtube(url, carpeta_audio)
    nombre_base = os.path.splitext(os.path.basename(audio_path))[0]

    texto = transcribir_audio(audio_path, modelo="tiny")
    guardar_texto(texto, carpeta_texto, nombre_base)

    segments = diarizar_audio(audio_path, num_speakers=num_speakers, max_duration=60)
    transcripciones = transcribir_segmentos(audio_path, segments, modelo="tiny")

    salida = os.path.join(carpeta_texto, f"{nombre_base}_con_speakers.txt")
    with open(salida, "w", encoding="utf-8") as f:
        for speaker, texto in transcripciones:
            f.write(f"{speaker}: {texto}\n")
    print(f"✅ Transcripción con identificación de hablantes guardada en: {salida}")