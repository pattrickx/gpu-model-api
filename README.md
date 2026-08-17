# GPU Model API

API FastAPI para **geração de imagens com GPU** usando modelos de difusão
(FLUX.1-schnell e um modelo secundário SDXL), com **1 request por vez** para
não sobrecarregar a GPU.

> Testado em: NVIDIA RTX 3060 12GB, host TrueNAS, deploy via `docker compose`.

### Limitacoes conhecidas (RTX 3060 12GB)
- **FLUX.1-schnell** roda nesta 12GB usando a receita da GTX 1650 4GB:
  `torch_dtype=bfloat16` (sem quantizacao) + `enable_sequential_cpu_offload()`
  + `enable_vae_tiling()`. O texto (T5) roda em CPU, entao cada geracao demora
  ~5-8 min no host — mas funciona e nao da OOM.
- **model2 / SDXL** funciona 100% (fp16, ~7-10GB VRAM com
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), e e mais rapido.
- **FLUX exige `HF_TOKEN`** valido (repo e *gated*).
- A imagem precisa de **`gcc` + `python3-dev`** (o `bitsandbytes`/`triton`
  compila dentro do container). Veja "Build da imagem" abaixo.

---

## O que faz

Endpoints:

| Método | Endpoint | Função |
|---|---|---|
| GET | `/health` | status + modelos carregados |
| GET | `/models` | ids e repos dos modelos |
| POST | `/generate` | gera imagem e **retorna o PNG direto** |

O body do `/generate` escolhe o modelo:

| Campo `model` | Modelo |
|---|---|
| `"flux"` | `FLUX.1-schnell` (Apache-2.0, few-step) |
| `"model2"` | `stable-diffusion-xl-base-1.0` (fp16) |

Cada request recebe um **prompt**, o **model** (`flux`|`model2`), a
**orientação** (`w` = wide 16:9 1024x576, `t` = vertical 9:16 576x1024), a
**quantidade de interações** (`num_inference_steps`, os "loops") e a `seed`.
O endpoint retorna a imagem PNG diretamente (FileResponse).

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
├── docker-compose.yml   # stack Portainer (GPU passthrough)
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

## Deploy via Docker / Portainer

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

(Opcional) o workflow `.github/workflows/build.yml` também builda no push,
mas pode falhar por espaco/rede no runner — o metodo acima e o caminho
garantido.

### 2. Pré-requisitos no host (TrueNAS / Linux GPU)
- NVIDIA Container Toolkit instalado.
- Spec CDI habilitada: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
- Docker capaz de usar `--gpus all`.

### 3. Portainer — adicionar Stack por repositório Git
1. Portainer → **Stacks → Add stack**.
2. **Build method:** `Repository`.
3. **Repository URL:** `https://github.com/pattrickx/gpu-model-api`
4. **Compose path:** `docker-compose.yml`
5. (Opcional) **Environmental variables:** `HF_TOKEN` se usar modelo gated.
6. **Deploy the stack.**

> ⚠️ Em stack Git, **não use `build: .`** (Portainer não tem docker context e
> falha com erro de build). Por isso a imagem é pré-buildada no GHCR.

O container sobe na porta `8000` e faz GPU passthrough (`deploy.resources`
+ `gpus: all` via compose v2.4+ / Portainer).

---

## Uso da API

### Health check
```bash
curl http://SEU_HOST:8000/health
# {"status":"ok","models_loaded":{"flux":false,"model2":false}}
```

### Gerar imagem FLUX.1-schnell wide (horizontal), 4 loops
```bash
curl -X POST http://SEU_HOST:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"flux","prompt":"um lago sereno ao por do sol, luz volumetrica","orientation":"w","num_inference_steps":4}'
# retorna o PNG direto (salve com -o saida.png)
```

### Gerar imagem FLUX.1-schnell vertical, 4 loops
```bash
curl -X POST http://SEU_HOST:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"flux","prompt":"um lago sereno ao por do sol, luz volumetrica","orientation":"t","num_inference_steps":4}' -o flux_vertical.png
```

### Gerar imagem SDXL (model2) wide, 4 loops
```bash
curl -X POST http://SEU_HOST:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"model2","prompt":"viajante solitario num penhasco ao amanhecer","orientation":"w","num_inference_steps":4}' -o sdxl_wide.png
```

> FLUX na 12GB usa `sequential_cpu_offload` (texto em CPU) → cada geracao
> leva ~5-8 min. SDXL e mais rapido. A API serializa 1 request por vez.

### Baixar a imagem gerada (pelo filename, sem GPU)
```bash
curl http://SEU_HOST:8000/image/<filename> -o saida.png
```

---

## Variáveis de ambiente

| Var | Default | Descrição |
|---|---|---|
| `HF_TOKEN` | — | Token HF (só p/ modelos gated) |
| `HF_HOME` | `/models` | Cache dos pesos (volume) |
| `MODEL_A_REPO` | `black-forest-labs/FLUX.1-schnell` | Modelo A |
| `MODEL_B_REPO` | `stabilityai/stable-diffusion-xl-base-1.0` | Modelo B |
| `DEFAULT_NUM_STEPS` | `4` | Loops padrão |
| `DEFAULT_SEED` | `42` | Seed padrão |
| `TORCH_DTYPE` | `bfloat16` | Precisão de carga |
| `OUTPUT_DIR` | `/app/output` | Onde salvar PNGs |

---

## Limitações conhecidas
- **FLUX.1 (dev/kontext) não roda em GPU de 12GB** (o transformer exige ~11.5GB
  só de cálculo). Por isso o modelo A usa **FLUX.1-schnell** (few-step) com
  4-bit + VAE bf16, que cabe. Se quiser FLUX.1-dev, use GPU com ≥24GB.
- Modelos são carregados **lazy** no primeiro request de cada modelo (pode levar
  minutos no primeiro uso enquanto baixa os pesos).
