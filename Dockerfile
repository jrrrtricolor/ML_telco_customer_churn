from python:3.13.12

#Metadados da imagem

LABEL maintainer="jrrrtricolor@gmail.com"
LABEL version="1.0"
LABEL description="Dockerfile para analise de churn da base de dados de Telecom."

#Variáveis de ambiente

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

#Diretorio de trabalho
WORKDIR /app

#Instalação das dependências
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar as dependências do projeto
COPY requirement.txt .
run pip install --upgrade pip
run pip install --no-cache-dir -r requirement.txt

# Copiar projeto todo para o container
copy . .

# Copiar outros arquivos necessários para o container
copy pyproject.toml .

# Criar usuário não-root para rodar a aplicação
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Porta que a aplicação irá rodar
EXPOSE 8000

# HEALTHCHECK para verificar se a aplicação está rodando
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para rodar a aplicação
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
