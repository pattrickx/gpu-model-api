#!/usr/bin/env python3
"""Cliente de teste da API expandida (roda DENTRO do container python-ssh).
Testa os 4 modelos com resultado valido: Kokoro, Qwen3-TTS, Whisper-turbo, SDXL.
Baixa/guarda os resultados em /workspace/gpu-api/results_test/."""
import json, urllib.request, urllib.error, os

BASE = "http://127.0.0.1:8000"
OUT = "/workspace/gpu-api/results_test"
os.makedirs(OUT, exist_ok=True)

def post_json(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + "/generate", data=data,
                                  headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)

def post_transcribe(audio_path, model="whisper_turbo", language="portuguese"):
    import requests  # fallback; se nao existir usa urllib multipart manual
    # usa urllib multipart simples
    boundary = "----hermesboundary"
    with open(audio_path, "rb") as f:
        body = f.read()
    # monta multipart
    import io
    buf = io.BytesIO()
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(audio_path)}"\r\n'.encode())
    buf.write(b"Content-Type: audio/wav\r\n\r\n")
    buf.write(body)
    buf.write(b"\r\n")
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    buf.write(model.encode())
    buf.write(b"\r\n")
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(b'Content-Disposition: form-data; name="language"\r\n\r\n')
    buf.write(language.encode())
    buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(BASE + "/transcribe", data=buf.getvalue(),
                                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return urllib.request.urlopen(req, timeout=600)

print("=== TESTE KOKORO (TTS pt-BR) ===")
try:
    r = post_json({"model": "kokoro", "text": "Olá, este é um teste de voz em português gerado pela API.", "voice": "af_heart"})
    print("status", r.status)
    # salva o wav
    fn = r.headers.get("Content-Disposition", "").split("filename=")[-1].strip('"') or "kokoro_out.wav"
    path = os.path.join(OUT, fn)
    with open(path, "wb") as f: f.write(r.read())
    print("salvo", path, os.path.getsize(path), "bytes")
except Exception as e:
    print("ERRO KOKORO:", repr(e))

print("=== TESTE QWEN3-TTS (TTS EN, Ryan) ===")
try:
    r = post_json({"model": "qwen3tts", "text": "She said she would be here by noon, but I haven't heard from her yet.", "voice": "Ryan", "language": "English"})
    print("status", r.status)
    fn = r.headers.get("Content-Disposition", "").split("filename=")[-1].strip('"') or "qwen_out.wav"
    path = os.path.join(OUT, fn)
    with open(path, "wb") as f: f.write(r.read())
    print("salvo", path, os.path.getsize(path), "bytes")
except Exception as e:
    print("ERRO QWEN:", repr(e))

print("=== TESTE WHISPER-TURBO (ASR) ===")
audio = "/workspace/bench/assets/whisper_in_synth.wav"
if os.path.exists(audio):
    try:
        r = post_transcribe(audio)
        txt = r.read().decode()
        path = os.path.join(OUT, "whisper_turbo_out.json")
        with open(path, "w") as f: f.write(txt)
        print("status", r.status, "salvo", path, os.path.getsize(path), "bytes")
        print("transcricao:", txt[:200])
    except Exception as e:
        print("ERRO WHISPER:", repr(e))
else:
    print("audio nao encontrado:", audio)

print("=== TESTE SDXL (imagem) ===")
try:
    r = post_json({"model": "model2", "prompt": "viajante solitario num penhasco ao amanhecer, fotografia 35mm", "orientation": "w", "num_inference_steps": 4})
    print("status", r.status)
    fn = r.headers.get("Content-Disposition", "").split("filename=")[-1].strip('"') or "sdxl_out.png"
    path = os.path.join(OUT, fn)
    with open(path, "wb") as f: f.write(r.read())
    print("salvo", path, os.path.getsize(path), "bytes")
except Exception as e:
    print("ERRO SDXL:", repr(e))

print("TESTES_CONCLUIDOS")
