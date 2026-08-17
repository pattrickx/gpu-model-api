# Imagem base CUDA com Python 3.12 + pip + torch + CUDA ja prontos.
# Evita apt-get (que falha na rede de alguns builders) e garante GPU.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias (torch ja vem na imagem base; nao reinstalamos).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo da aplicacao
COPY app ./app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
