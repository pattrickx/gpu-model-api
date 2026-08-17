# GPU Model API

API FastAPI para **geração de imagens com GPU** usando modelos de difusão
(FLUX.1-schnell e um modelo secundário SDXL), com **1 request por vez** para
não sobrecarregar a GPU.

> Testado em: NVIDIA RTX 3060 12GB, host TrueNAS, deploy via Portainer (stack Git).

---

## O que faz

| Endpoint | Modelo (modelo A) | Modelo (modelo B) |
|---|---|---|
| `POST /generate/flux` | `FLUX.1-schnell` (Apache-2.0, few-step) | — |
| `POST /generate/model2` | — | `stable-diffusion-xl-base-1.0` (fp16) |

Cada request recebe um **prompt**, a **orientação** (`w` = wide 16:9 1024x576,
`t` = vertical 9:16 576x1024), a **quantidade de interações** (`num_inference_steps`,
os "loops") e a `seed`. A imagem é salva em `/app/output` e pode ser baixada.

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

### 1. Build da imagem (automático via GitHub Actions → GHCR)
Ao fazer `push` para `main`, o workflow `.github/workflows/build.yml` builda e
publica `ghcr.io/pattrickx/gpu-model-api:latest` (público).

Ou build manual:
```bash
docker build -t ghcr.io/pattrickx/gpu-model-api:latest .
docker push ghcr.io/pattrickx/gpu-model-api:latest
```

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

### Gerar imagem wide (FLUX.1-schnell), 4 loops
```bash
curl -X POST http://SEU_HOST:8000/generate/flux \
  -H "Content-Type: application/json" \
  -d '{"prompt":"um lago sereno ao por do sol, luz volumetrica","orientation":"w","num_inference_steps":4}'
```

### Gerar imagem vertical (modelo 2 / SDXL), 6 loops
```bash
curl -X POST http://SEU_HOST:8000/generate/model2 \
  -H "Content-Type: application/json" \
  -d '{"prompt":"viajante solitario num penhasco ao amanhecer","orientation":"t","num_inference_steps":6,"seed":7}'
```

### Baixar a imagem gerada
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
