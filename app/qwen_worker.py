#!/usr/bin/env python3
"""qwen_worker.py — gera audio TTS com Qwen3-TTS num processo separado.

O Qwen3TTSModel deadlocka o CUDA context quando carregado DENTRO do processo
do uvicorn (qualquer thread). Rodar num subprocesso isolado (sua propria thread
principal) funciona, como comprovado por testes isolados.

Uso: python qwen_worker.py <texto> <voz> <idioma> <arquivo_saida>
"""
import sys
import os

# garante que este processo use a GPU corretamente
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

text = sys.argv[1]
voice = sys.argv[2] if len(sys.argv) > 2 else "Ryan"
language = sys.argv[3] if len(sys.argv) > 3 else "English"
out_path = sys.argv[4] if len(sys.argv) > 4 else "/workspace/gpu-api/results_test/qwen_out.wav"

from qwen_tts import Qwen3TTSModel
import torch
import soundfile as sf

repo = os.environ.get("MODEL_D_REPO", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
model = Qwen3TTSModel.from_pretrained(
    repo,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)
wavs, sr = model.generate_custom_voice(
    text=[text],
    language=[language],
    speaker=[voice],
)
sf.write(out_path, wavs[0], sr)
print(f"QWEN_WORKER_OK {out_path} {sr} {len(wavs[0])}")
