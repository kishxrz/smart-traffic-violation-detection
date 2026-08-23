# ============================================================
# Smart Traffic AI — Root Production Dockerfile for Render / Cloud
# ============================================================

# ── Stage 1: Dependencies ────────────────────────────────────
FROM python:3.13-slim AS deps

WORKDIR /install

COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --prefix=/opt/venv -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.13-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

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

# Expose ports
EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, os; port=os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/api/health')"

# Start Uvicorn server on $PORT or fallback to 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
