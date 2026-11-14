# Runtime image for the moderation API.
#
# The ONNX model is baked into the image rather than downloaded at startup, so
# the container starts without network access and always serves the exact
# version that was verified.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Keep the BLAS/OpenMP pools out of the way; ONNX Runtime's own thread count
# (MOD_ONNX_THREADS) is the only thing we tune.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only the three runtime artifacts, so a stray float graph in the build
# context can never end up in the image.
COPY models/model_int8.onnx ./models/model_int8.onnx
COPY models/tokenizer.json ./models/tokenizer.json
COPY models/model_meta.json ./models/model_meta.json

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"]

# Workers are capped so that workers * intra-op threads stays inside the core
# count of the target VPS.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--no-access-log"]
