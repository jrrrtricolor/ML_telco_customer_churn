FROM python:3.13.12

LABEL maintainer="jrrrtricolor@gmail.com"
LABEL version="1.0"
LABEL description="Dockerfile para analise de churn da base de dados de Telecom."

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirement.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirement.txt

COPY . .

# Gera mlflow.db e mlruns/modelos dentro da imagem para a API iniciar sem passos manuais.
RUN python -m src.main

RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
