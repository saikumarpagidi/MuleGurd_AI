# services/ai-engine/Dockerfile
#
# Multi-stage build: keep the runtime image slim by not shipping build tools
# (gcc, headers needed to compile lightgbm/torch wheels on some platforms).

FROM python:3.11-slim AS builder

WORKDIR /build
COPY services/ai-engine/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

# libgomp1 is required at runtime by LightGBM's OpenMP-parallelized inference.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY services/ai-engine/app/ ./app/

# Model artifacts — in production these SHOULD be mounted as a volume or
# pulled from an artifact store at container start rather than baked into
# the image, so a model update doesn't require a full image rebuild.
COPY ml/models/ ./ml/models/

EXPOSE 8002

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8002/health', timeout=2).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
