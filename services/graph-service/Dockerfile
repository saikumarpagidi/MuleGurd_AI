# Graph Service — Dockerfile
# Sprint 5 v2: Real TGN inference + Neo4j driver
FROM python:3.11-slim-bookworm

WORKDIR /app

# libgomp1: required by PyTorch CPU build (OpenMP runtime for parallel tensor ops)
# libgfortran5: required by scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only wheel first (avoids pulling the 2GB CUDA wheel)
# The --extra-index-url approach is the official recommended method per PyTorch docs
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric after torch is available
RUN pip install --no-cache-dir \
    torch-geometric==2.5.3

# Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY app/ ./app/



EXPOSE 8003

# start-period=60s to allow TGN checkpoint load (~15s for 1.49GB)
HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", "--log-level", "info"]
