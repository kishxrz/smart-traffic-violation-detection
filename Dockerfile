# ============================================================
# Smart Traffic AI — Root Production Dockerfile for Render / Cloud
# ============================================================

# ── Stage 1: Dependencies ────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /install

COPY backend/requirements.txt ./requirements.txt

# Install numpy 1.26.4 and CPU PyTorch in one consistent step to avoid numpy 2.x version mixing
RUN pip install --no-cache-dir --prefix=/opt/venv "numpy==1.26.4" torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/opt/venv -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8000

# Install OpenCV system dependencies and ffmpeg non-interactively
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built packages from the deps stage
COPY --from=deps /opt/venv /usr/local

# Copy application code and model weights from repository root context
COPY backend/app/ ./app/
COPY models/ ./models/

# Create evidence directories
RUN mkdir -p evidence models /tmp/evidence

# Create a non-root user for security
RUN useradd -m -u 1000 trafficai && chown -R trafficai:trafficai /app /tmp/evidence
USER trafficai

# Expose port 8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Start Uvicorn server on port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
