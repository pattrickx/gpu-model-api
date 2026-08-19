#!/usr/bin/env python3
"""Teste incremental dos endpoints da API (unbuffered). Roda DENTRO do container."""
import json, os, io, urllib.request, sys

BASE = "http://127.0.0.1:8080"
OUT = "/workspace/gpu-api/results_test"
os.makedirs(OUT, exist_ok=True)
def log(m): print(m, flush=True)

def post_json(payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + "/generate", data=data,
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def post_transcribe(audio_path, model="whisper_turbo", language="portuguese", timeout=600):
    boundary = "----hb"
    with open(audio_path, "rb") as f: body = f.read()
    buf = io.BytesIO()
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(audio_path)}"\r\n'.encode())
    buf.write(b"Content-Type: audio/wav\r\n\r\n")
    buf.write(body); buf.write(b"\r\n")
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(b'Content-Disposition: form-data; name="model"\r\n\r\n'); buf.write(model.encode()); buf.write(b"\r\n")
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(b'Content-Disposition: form-data; name="language"\r\n\r\n'); buf.write(language.encode()); buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(BASE + "/transcribe", data=buf.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return urllib.request.urlopen(req, timeout=timeout)

log("=== KOKORO ===")
try:
    r = post_json({"model":"kokoro","text":"Olá, este é um teste de voz em português gerado pela API.","voice":"af_heart"})
    fn = r.headers.get("Content-Disposition","").split("filename=")[-1].strip('"') or "kokoro.wav"
    p = os.path.join(OUT, fn)
    open(p,"wb").write(r.read())
    log(f"KOKORO OK {os.path.getsize(p)} bytes -> {fn}")
except Exception as e:
    log(f"KOKORO ERRO: {repr(e)}")

log("=== QWEN3-TTS ===")
try:
    r = post_json({"model":"qwen3tts","text":"She said she would be here by noon, but I haven't heard from her yet.","voice":"Ryan","language":"English"})
    fn = r.headers.get("Content-Disposition","").split("filename=")[-1].strip('"') or "qwen.wav"
    p = os.path.join(OUT, fn)
    open(p,"wb").write(r.read())
    log(f"QWEN OK {os.path.getsize(p)} bytes -> {fn}")
except Exception as e:
    log(f"QWEN ERRO: {repr(e)}")

log("=== WHISPER-TURBO ===")
audio = "/workspace/bench/assets/whisper_in_synth.wav"
if os.path.exists(audio):
    try:
        r = post_transcribe(audio)
        txt = r.read().decode()
        p = os.path.join(OUT, "whisper_turbo_out.json")
        open(p,"w").write(txt)
        log(f"WHISPER OK {os.path.getsize(p)} bytes -> {p}")
        log("transcricao: " + txt[:150])
    except Exception as e:
        log(f"WHISPER ERRO: {repr(e)}")
else:
    log(f"WHISPER skip (audio ausente: {audio})")

log("=== SDXL ===")
try:
    r = post_json({"model":"model2","prompt":"viajante solitario num penhasco ao amanhecer, fotografia 35mm","orientation":"w","num_inference_steps":4})
    fn = r.headers.get("Content-Disposition","").split("filename=")[-1].strip('"') or "sdxl.png"
    p = os.path.join(OUT, fn)
    open(p,"wb").write(r.read())
    log(f"SDXL OK {os.path.getsize(p)} bytes -> {fn}")
except Exception as e:
    log(f"SDXL ERRO: {repr(e)}")

log("TESTES_CONCLUIDOS")
