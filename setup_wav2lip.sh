#!/usr/bin/env bash
# setup_wav2lip.sh — prepara o Wav2Lip no container da gpu-model-api.
# Chamado pelo docker-compose.yml (command) ANTES do uvicorn.
# Clona o repo, baixa o checkpoint Wav2Lip-GAN (435MB, mirror HF) e instala deps.
set -e

WAV2LIP_DIR=/app/wav2lip
CHECKPOINT=$WAV2LIP_DIR/checkpoints/wav2lip_gan.pth

echo "[setup_wav2lip] iniciando..."

if [ ! -d "$WAV2LIP_DIR/.git" ]; then
  echo "[setup_wav2lip] clonando repo..."
  rm -rf "$WAV2LIP_DIR"
  git clone -q https://github.com/Rudrabha/Wav2Lip.git "$WAV2LIP_DIR"
fi

echo "[setup_wav2lip] instalando deps (librosa 0.9.2 compativel)..."
pip install -q "librosa==0.9.2" face_alignment insightface tqdm av 2>&1 | tail -1 || true

if [ ! -f "$CHECKPOINT" ]; then
  echo "[setup_wav2lip] baixando checkpoint Wav2Lip-GAN (mirror HF)..."
  mkdir -p "$WAV2LIP_DIR/checkpoints"
  python3 - <<'PY'
import urllib.request, os
ckpt = "/app/wav2lip/checkpoints/wav2lip_gan.pth"
url = "https://huggingface.co/rippertnt/wav2lip/resolve/main/checkpoints/wav2lip_gan.pth?download=true"
print("download start")
urllib.request.urlretrieve(url, ckpt)
print("download done", os.path.getsize(ckpt))
PY
fi

echo "[setup_wav2lip] pronto: $(ls -la $CHECKPOINT 2>/dev/null | awk '{print $5}') bytes"
