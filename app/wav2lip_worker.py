#!/usr/bin/env python3
"""wav2lip_worker.py — subprocesso isolado para rodar o Wav2Lip.

O Wav2Lip depende de librosa==0.9.2, face_alignment, insightface e opencv,
que podem conflitar com o ambiente principal. Roda em subprocesso para
isolamento. Recebe JSON via argv[1] e imprime JSON no stdout.

Args (JSON):
  face_path   : caminho do vídeo OU imagem de entrada (face)
  audio_path  : caminho do WAV de áudio
  out_path    : caminho de saída do vídeo
  pads        : str "top right bottom left" (default "0 20 0 0")
  resize_factor: int (default 1)
"""
import json
import os
import subprocess
import sys

WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", "/app/wav2lip")


def main() -> int:
    req = json.loads(sys.argv[1])
    face_path = req["face_path"]
    audio_path = req["audio_path"]
    out_path = req["out_path"]
    pads = req.get("pads", "0 20 0 0")
    resize_factor = req.get("resize_factor", 1)

    # Debug: garante que os inputs existem no container.
    if not os.path.exists(face_path):
        sys.stderr.write(f"WORKER_ERR: face_path inexistente: {face_path}\n")
        return 2
    if not os.path.exists(audio_path):
        sys.stderr.write(f"WORKER_ERR: audio_path inexistente: {audio_path}\n")
        return 2

    # O Wav2Lip escreve saidas relativas (temp/result.avi) no cwd, entao
    # precisamos rodar a partir do diretorio do repo.
    os.chdir(WAV2LIP_DIR)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        "python3",
        os.path.join(WAV2LIP_DIR, "inference.py"),
        "--checkpoint_path",
        os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth"),
        "--face",
        face_path,
        "--audio",
        audio_path,
        "--outfile",
        out_path,
        "--pads",
        *str(pads).split(),
        "--resize_factor",
        str(resize_factor),
    ]

    # Wav2Lip escreve logs em stderr; capturamos para o stdout do worker.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return 1
    if not os.path.exists(out_path):
        sys.stderr.write("Wav2Lip nao produziu saida: " + proc.stderr[-500:])
        return 1
    print(json.dumps({"out_path": out_path, "size": os.path.getsize(out_path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
