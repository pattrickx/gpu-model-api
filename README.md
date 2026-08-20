# GPU Model API

API FastAPI para **geração de imagens, áudio (TTS) e transcrição de áudio (ASR)**
com GPU, com **1 request por vez** para não sobrecarregar a GPU.

Endpoints separados por tipo de mídia:

- `/generate-image` — imagem (FLUX.1-schnell, SDXL)
- `/generate-audio` — TTS (Kokoro-82M, Qwen3-TTS, Chatterbox, F5-TTS*)
- `/transcribe-audio` — ASR (Whisper, CrisperWhisper, Parakeet, Qwen3-ASR)
- `/generate-lipsync` — Lipsync (Wav2Lip: vídeo|imagem + áudio → MP4)

> Testado em: NVIDIA RTX 3060 12GB, host TrueNAS, deploy via `docker compose`.

### Limitações conhecidas (RTX 3060 12GB)
- **FLUX.1-schnell** roda nesta 12GB usando a receita da GTX 1650 4GB:
  `torch_dtype=bfloat16` (sem quantizacao) + `enable_sequential_cpu_offload()`
  + `enable_vae_tiling()`. O texto (T5) roda em CPU, entao cada geracao demora
  ~5-8 min no host — mas funciona e nao da OOM.
- **model2 / SDXL** funciona 100% (fp16, ~7-10GB VRAM com
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), e e mais rapido.
- **FLUX exige `HF_TOKEN`** valido (repo e *gated*).
- A imagem precisa de **`gcc` + `python3-dev`** (o `bitsandbytes`/`triton`
  compila dentro do container). Veja "Deploy via Docker" abaixo.
- O **ASR transcreve sempre na língua original** do áudio: o parâmetro
  `language` é ignorado de propósito (o modelo detecta o idioma sozinho).

---

## O que faz

Endpoints:

| Método | Endpoint | Função |
|---|---|---|
| GET | `/health` | status + modelos carregados |
| GET | `/models` | ids e repos dos modelos |
| GET | `/docs` | Swagger UI interativo |
| POST | `/generate-image` | gera imagem (T2I), retorna PNG |
| POST | `/generate-image-edit` | edita imagem (texto+imagem → PNG) via FLUX.2-klein |
| POST | `/generate-audio` | gera áudio (TTS), retorna WAV |
| POST | `/generate-lipsync` | lipsync (Wav2Lip: vídeo|imagem + áudio → MP4) |
| POST | `/transcribe-audio` | transcreve áudio, retorna JSON |

Cada endpoint escolhe o modelo via campo `model`. Os campos por tipo:
- **Imagem** (`/generate-image`): `model`
  (`flux`|`model2`|`sdxl_turbo`|`sd_turbo`|`sdxl_lightning`|`flux2_klein`|
  `flux2_klein_fp8`|`flux2_klein_base`), `prompt`, `orientation`
  (`w`=wide 16:9 1024x576, `t`=vertical 9:16 576x1024; os turbo forçam 512x512),
  `num_inference_steps`, `seed`.
- **Edição de imagem** (`/generate-image-edit`): `image` (arquivo, multipart),
  `model` (`flux2_klein`|`flux2_klein_fp8`|`flux2_klein_base`), `prompt`
  (prompt de edição), `orientation`, `num_inference_steps`, `seed`. O
  FLUX.2-klein unifica T2I e I2I: a imagem de entrada é passada como
  condição (sem `strength` — edição in-painting-style).
- **Áudio/TTS** (`/generate-audio`): `model` (`kokoro`|`qwen3tts`|`chatterbox`|`f5tts`), `text`,
  `voice`, `language`, `exaggeration` (chatterbox), `cfg_weight` (chatterbox),
  `ref_audio` (f5tts), `ref_text` (f5tts). O f5tts é **CC-BY-NC** (não comercial)
  e English-only (voice cloning via áudio de referência).
- **ASR** (`/transcribe-audio`): `file` (multipart WAV), `model`
  (whisper_turbo|crisper2|parakeet|qwen3asr_06b|qwen3asr_17b). O `language`
  é **ignorado** (detecção automática).

O `/generate-image` e `/generate-audio` retornam o arquivo direto
(FileResponse). O `/transcribe-audio` retorna
`{"model": "...", "transcription": "{...json com text + chunks/language...}"}`.

### Concorrência (regra de ouro)
A GPU é serial por natureza. **Todas as gerações passam por um executor de 1
worker + lock global**, então o segundo request só entra quando o primeiro
termina. Chamadas concorrentes **enfileiram** (não falham). Isso evita OOM /
travamento da GPU.

---

## Estrutura

```
gpu-model-api/
├── app/
│   ├── __init__.py
│   ├── config.py        # variaveis de ambiente + resolucao por orientacao
│   ├── schemas.py       # Pydantic (request/response)
│   ├── pipelines.py     # carga lazy + lock serializado (1 req/vez)
│   └── main.py          # FastAPI (endpoints)
├── docker-compose.yml   # compose local (GPU passthrough), bind mount RO
├── Dockerfile           # base nvidia/cuda:12.4, Python 3.12
├── requirements.txt
├── .env.example
└── README.md
```

---

## Como rodar localmente (fora do Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME=/caminho/para/cache
export HF_TOKEN=seu_token   # so p/ modelos gated
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Deploy via Docker (compose + bind mount read-only)

O deploy usa **`docker compose`** (não stack Portainer) com **bind mount
read-only** do código e **1 executor serial**. O container tem `restart:
unless-stopped`, então sobrevive a reboot.

### 1. Build da imagem (GHCR)
A imagem `ghcr.io/pattrickx/gpu-model-api:latest` ja esta publicada. Para
rebuildar (ex.: mudar versoes), o `docker build` falha na rede do TrueNAS
(buildkit isola a rede nas etapas RUN). Use o metodo que funciona:

```bash
# No host com Docker + GPU e rede liberada (--network host):
CID=$(docker run -d --network host -v $PWD:/build \
  nvidia/cuda:12.4.0-runtime-ubuntu22.04 bash -c "
  apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential git && \
  pip3 install --no-cache-dir -r /build/requirements.txt && \
  mkdir -p /app && cp -r /build/app /app && echo BUILD_OK")
docker wait $CID
docker commit $CID ghcr.io/pattrickx/gpu-model-api:latest
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u pattrickx --password-stdin
docker push ghcr.io/pattrickx/gpu-model-api:latest
```

### 2. Pré-requisitos no host (TrueNAS / Linux GPU)
- NVIDIA Container Toolkit instalado.
- Spec CDI habilitada: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
- Docker capaz de usar `--gpus all`.

### 3. Subir a stack (compose)
O `docker-compose.yml` faz bind mount **read-only** do código
(`/root/build/gpu-model-api/app:/app/app:ro`) e instala as libs extras no
`command` (python-multipart, imageio-ffmpeg, kokoro, soundfile, qwen-asr,
torchaudio). Para subir:

```bash
cd /root/gpu-model-api-deploy
docker compose up -d
```

O container sobe na porta `8000` e faz GPU passthrough (`deploy.resources`
+ `gpus: all`). Edições no código exigem `docker compose restart` (uvicorn
sem `--reload`).

---

## Uso da API

### Health check
```bash
curl http://SEU_HOST:8000/health
# {"status":"ok","models_loaded":{"flux":false,"model2":false,...}}
```

### Listar modelos
```bash
curl http://SEU_HOST:8000/models
# {"flux":{"repo":"...","task":"image"}, ...}
```

---

### Imagem — FLUX.1-schnell wide (horizontal), 4 loops
```bash
curl -X POST http://SEU_HOST:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{"model":"flux","prompt":"um lago sereno ao por do sol, luz volumetrica","orientation":"w","num_inference_steps":4}' -o flux_wide.png
```

### Imagem — FLUX.1-schnell vertical, 4 loops
```bash
curl -X POST http://SEU_HOST:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{"model":"flux","prompt":"um lago sereno ao por do sol, luz volumetrica","orientation":"t","num_inference_steps":4}' -o flux_vertical.png
```

### Imagem — SDXL (model2) wide, 4 loops
```bash
curl -X POST http://SEU_HOST:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{"model":"model2","prompt":"viajante solitario num penhasco ao amanhecer","orientation":"w","num_inference_steps":4}' -o sdxl_wide.png
```

### Imagem — FLUX.2-klein-4B (bf16, offload) wide, 4 steps
```bash
curl -X POST http://SEU_HOST:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{"model":"flux2_klein","prompt":"viajante solitario num penhasco ao amanhecer","orientation":"w","num_inference_steps":4}' -o klein_wide.png
```

### Imagem — SDXL-Turbo (512x512 fixo, mais rapido) 4 steps
```bash
curl -X POST http://SEU_HOST:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{"model":"sdxl_turbo","prompt":"viajante solitario num penhasco ao amanhecer","num_inference_steps":4}' -o sdxl_turbo.png
```

> `sdxl_turbo`/`sd_turbo` ignoram `orientation` e forçam 512x512 (checkpoint
> treinado só nessa resolução). `sd_turbo` é o mais leve (~3GB VRAM, ~2s).

### Edição de imagem — FLUX.2-klein I2I (texto + imagem → PNG)
```bash
curl -X POST http://SEU_HOST:8000/generate-image-edit \
  -F "image=@entrada.png" \
  -F "model=flux2_klein" \
  -F "prompt=a mesma cena, porem a noite com iluminacao neon" \
  -F "num_inference_steps=4" -o editada.png
```

> O `/generate-image-edit` aceita um arquivo de imagem (`image`) + prompt de
> edição. Modelos: `flux2_klein` (bf16), `flux2_klein_fp8` (fp8 dequantizado),
> `flux2_klein_base` (não-destilado, use ~20 steps e guidance>0). O FLUX.2-klein
> unifica T2I e I2I num pipeline; não use `strength` (edição in-painting-style).
>
> Testado: `POST /generate-image-edit` com `flux2_klein`, `flux2_klein_fp8` e
> `flux2_klein_base` retornou PNG 1024x576 válido (edição de uma imagem base
> gerada por T2I). Ver `test-gpu-api/results-models/image2image-*`.

> FLUX na 12GB usa `sequential_cpu_offload` (texto em CPU) → cada geracao
> leva ~5-8 min. SDXL e mais rapido. A API serializa 1 request por vez.

### Áudio/TTS — Kokoro-82M (texto -> WAV, PT-BR)
```bash
curl -X POST http://SEU_HOST:8000/generate-audio \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","text":"Olá, este é um teste de voz em português.","voice":"af_heart"}' -o kokoro.wav
```

### Áudio/TTS — Qwen3-TTS (vozes nomeadas EN/PT)
```bash
curl -X POST http://SEU_HOST:8000/generate-audio \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3tts","text":"She said she would be here by noon.","voice":"Ryan","language":"English"}' -o qwen_ryan.wav
```

### Áudio/TTS — Chatterbox (ResembleAI, 23+ línguas incl. PT-BR, MIT)
```bash
curl -X POST http://SEU_HOST:8000/generate-audio \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox","text":"Olá, este é um teste de síntese de voz em português brasileiro.","exaggeration":0.5,"cfg_weight":0.5}' -o chatterbox.wav
```

> `exaggeration` (0.25–2.0, default 0.5) controla expressividade; `cfg_weight`
> (0.0–1.0, default 0.5) controla aderência ao texto. Modelo 0.5B, ~24–53s
> por áudio na 3060 12GB. RoDA em PT-BR (testado).

### Áudio/TTS — F5-TTS (SWivid, Flow Matching, **CC-BY-NC**, English-only)
```bash
curl -X POST http://SEU_HOST:8000/generate-audio \
  -H "Content-Type: application/json" \
  -d '{"model":"f5tts","text":"The quick brown fox jumps over the lazy dog.","ref_text":"This is a reference voice sample used for voice cloning."}' -o f5tts.wav
```
> `ref_audio` (caminho no container, default `/app/ref_audio_en.wav`) e `ref_text`
> definem a voz clonada. English-only. Licença **CC-BY-NC-4.0 (não comercial)** —
> integrado para prova de conceito. RoDA em ~9s/áudio (24kHz).

### Lipsync — Wav2Lip (vídeo|imagem + áudio → MP4)
```bash
# Modo vídeo:
curl -X POST http://SEU_HOST:8000/generate-lipsync \
  -F "face=@video.mp4" -F "audio=@audio.wav" -o lipsync.mp4
# Modo imagem (foto → vídeo sincronizado):
curl -X POST http://SEU_HOST:8000/generate-lipsync \
  -F "face=@foto.jpg" -F "audio=@audio.wav" -o lipsync.mp4
```
> `face` = vídeo (.mp4) OU imagem (.jpg/.png); `audio` = WAV. O vídeo de saída
> assume a duração do áudio (Wav2Lip estica repetindo frames quando áudio > vídeo).
> `pads` (`"0 20 0 0"`) ajusta o crop do rosto. Checkpoint Wav2Lip-GAN (435MB)
> baixado via `setup_wav2lip.sh` no startup do container.

### ASR — Whisper-large-v3-turbo (língua original, word-level)
```bash
curl -X POST http://SEU_HOST:8000/transcribe-audio \
  -F "file=@audio.wav" -F "model=whisper_turbo"
# retorna {"model":"whisper_turbo","transcription":"{...json com text + chunks...}"}
```

### ASR — CrisperWhisper2.0_large (timestamps precisos de palavra)
```bash
curl -X POST http://SEU_HOST:8000/transcribe-audio \
  -F "file=@audio.wav" -F "model=crisper2"
# retorna JSON com text + chunks (word timestamps)
```

### ASR — Parakeet TDT 0.6B V3 (inglês, word/char/segment nativos)
```bash
curl -X POST http://SEU_HOST:8000/transcribe-audio \
  -F "file=@audio.wav" -F "model=parakeet"
# transcreve em ingles (monolingue); timestamps nativos do TDT
```

### ASR — Qwen3-ASR-0.6B (multilíngue, word timestamps via ForcedAligner)
```bash
curl -X POST http://SEU_HOST:8000/transcribe-audio \
  -F "file=@audio.wav" -F "model=qwen3asr_06b"
# detecta idioma automaticamente; chunks com start_time/end_time por palavra
```

### ASR — Qwen3-ASR-1.7B (multilíngue, word timestamps via ForcedAligner)
```bash
curl -X POST http://SEU_HOST:8000/transcribe-audio \
  -F "file=@audio.wav" -F "model=qwen3asr_17b"
# detecta idioma automaticamente (inclui pt); chunks com start_time/end_time
```

> Todos os ASR acima transcrevem na **língua original** do áudio (o parâmetro
> `language` é ignorado de propósito). Use `jq` para inspecionar o JSON:
> `curl -s ... | jq -r '.transcription' | jq`.

### Baixar o arquivo gerado (pelo filename, sem GPU)
```bash
curl http://SEU_HOST:8000/file/<filename> -o saida.ext
```

---

## Modelos disponíveis

| `model` | Tipo | Repo | Saída |
|---|---|---|---|
| `flux` | imagem | `MODEL_A_REPO` (FLUX.1-schnell) | PNG |
| `model2` | imagem | `MODEL_B_REPO` (SDXL base 1.0) | PNG |
| `kokoro` | TTS | `MODEL_C_REPO` (hexgrad/Kokoro-82M) | WAV |
| `qwen3tts` | TTS | `MODEL_D_REPO` (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | WAV |
| `chatterbox` | TTS | `MODEL_P_REPO` (ResembleAI/chatterbox) | WAV |
| `f5tts` | TTS | `MODEL_Q_REPO` (SWivid/F5-TTS, **CC-BY-NC**) | WAV |
| `whisper_turbo` | ASR | `MODEL_E_REPO` (openai/whisper-large-v3-turbo) | JSON |
| `crisper2` | ASR | `MODEL_F_REPO` (nyralabs/CrisperWhisper2.0_large) | JSON |
| `parakeet` | ASR | `MODEL_G_REPO` (nvidia/parakeet-tdt-0.6b-v3) | JSON |
| `qwen3asr_06b` | ASR | `MODEL_H_REPO` (Qwen/Qwen3-ASR-0.6B) + ForcedAligner | JSON |
| `qwen3asr_17b` | ASR | `MODEL_I_REPO` (Qwen/Qwen3-ASR-1.7B) + ForcedAligner | JSON |
| `sdxl_turbo` | imagem | `MODEL_J_REPO` (stabilityai/sdxl-turbo) | PNG |
| `sd_turbo` | imagem | `MODEL_K_REPO` (stabilityai/sd-turbo) | PNG |
| `sdxl_lightning` | imagem | `MODEL_L_REPO` (ByteDance/SDXL-Lightning) | PNG |
| `flux2_klein` | imagem | `MODEL_M_REPO` (black-forest-labs/FLUX.2-klein-4B) | PNG |
| `flux2_klein_fp8` | imagem | `MODEL_N_REPO` (black-forest-labs/FLUX.2-klein-4b-fp8) | PNG |
| `flux2_klein_base` | imagem | `MODEL_O_REPO` (black-forest-labs/FLUX.2-klein-base-4B) | PNG |
| `flux2_klein` / `flux2_klein_fp8` / `flux2_klein_base` | **edição** (`/generate-image-edit`) | mesmos repos acima | PNG |

> Os 3 `flux2_klein*` também fazem **edição de imagem** via
> `/generate-image-edit` (texto + imagem → PNG). O `unsloth/FLUX.2-klein-4B-GGUF`
> **não** foi integrado: exige loader `flux-gguf` que não existe no PyPI nem no
> diffusers 0.39 (bloqueio de ambiente, não do modelo).

Todos os modelos são carregados **lazy** no primeiro request e permanecem na
GPU; a concorrência é serializada (1 request por vez) para não sobrecarregar.

> `nyralabs/CrisperWhisper` (repo menor) **não** foi integrado: exige
> `tokenizers>=0.21,<0.22` (conflito com a versão instalada). Use `crisper2`
> (CrisperWhisper2.0_large), que funciona.

---

## Variáveis de ambiente

| Var | Default | Descrição |
|---|---|---|
| `HF_TOKEN` | — | Token HF (só p/ modelos gated) |
| `HF_HOME` | `/models` | Cache dos pesos (volume) |
| `MODEL_A_REPO` | `black-forest-labs/FLUX.1-schnell` | Imagem A |
| `MODEL_B_REPO` | `stabilityai/stable-diffusion-xl-base-1.0` | Imagem B |
| `MODEL_C_REPO` | `hexgrad/Kokoro-82M` | TTS Kokoro |
| `MODEL_D_REPO` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | TTS Qwen3 |
| `MODEL_E_REPO` | `openai/whisper-large-v3-turbo` | ASR Whisper |
| `MODEL_F_REPO` | `nyralabs/CrisperWhisper2.0_large` | ASR CrisperWhisper |
| `MODEL_G_REPO` | `nvidia/parakeet-tdt-0.6b-v3` | ASR Parakeet |
| `MODEL_H_REPO` | `Qwen/Qwen3-ASR-0.6B` | ASR Qwen3-ASR-0.6B |
| `MODEL_I_REPO` | `Qwen/Qwen3-ASR-1.7B` | ASR Qwen3-ASR-1.7B |
| `DEFAULT_NUM_STEPS` | `4` | Loops padrão |
| `DEFAULT_SEED` | `42` | Seed padrão |
| `TORCH_DTYPE` | `bfloat16` | Precisão de carga |
| `OUTPUT_DIR` | `/app/output` | Onde salvar arquivos |

---

## Limitações conhecidas
- **FLUX.1 (dev/kontext) não roda em GPU de 12GB** (o transformer exige ~11.5GB
  só de cálculo). Por isso o modelo A usa **FLUX.1-schnell** (few-step) com
  4-bit + VAE bf16, que cabe. Se quiser FLUX.1-dev, use GPU com ≥24GB.
- Modelos são carregados **lazy** no primeiro request de cada modelo (pode levar
  minutos no primeiro uso enquanto baixa os pesos).
- Os ASR **Qwen3-ASR** carregam junto o `Qwen3-ForcedAligner-0.6B` para word
  timestamps; isso ocupa VRAM extra (~1-3GB) enquanto ativos.
