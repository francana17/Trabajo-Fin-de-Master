# ---------- BASE ----------
# Usamos el mirror de Google para evitar timeouts de Docker Hub
FROM mirror.gcr.io/library/python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Dependencias de sistema para Prophet/CmdStan y numéricas (sin atlas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ gcc make git \
    libopenblas-dev liblapack-dev \
    libgomp1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias Python
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# Instalar CmdStan dentro de la imagen (evita compilar en Cloud Run)
RUN python -c "import cmdstanpy as c; c.install_cmdstan()"

# Copiar código
COPY . /app

# Config Streamlit para Cloud Run / Docker
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_HEADLESS=true

# Salud (opcional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/_stcore/health || exit 1

EXPOSE 8080
CMD ["streamlit", "run", "app.py"]
