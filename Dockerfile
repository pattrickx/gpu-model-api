# Imagem base CUDA (runtime) + Python 3.10 + pip instalado via apt.
# O build e' feito via `docker run --network host` + `docker commit` no host
# (o buildkit do TrueNAS isola a rede nas etapas RUN, entao o apt/pip falha
# em `docker build`). Esta Dockerfile documenta e reproduz o ambiente que
# funciona: gcc + python3-dev sao OBRIGATORIOS para o bitsandbytes/triton
# (usado no FLUX 4-bit) compilar dentro do container.
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# gcc + python3-dev: necessarios para o triton (bitsandbytes) compilar.
# python3-pip: instala o pip (a imagem base nao tem).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
