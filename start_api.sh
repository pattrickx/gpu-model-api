#!/bin/bash
# start_api.sh — sobe a GPU Model API (modo expandido: imagem+TTS+ASR) no venv do python-ssh.
# ATENCAO: defina HF_TOKEN via ambiente ou .env (NAO comite o token).
export HF_HOME=/workspace/bench/.hf_cache
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/gpu-api
source /workspace/venv/bin/activate
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > /workspace/gpu-api.log 2>&1 &
echo "API_PID=$!"
