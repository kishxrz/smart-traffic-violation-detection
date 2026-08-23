# ============================================================
# Smart Traffic AI — Backend Dockerfile
# Multi-stage build for lean production deployment (Local / HuggingFace / Cloud)
# ============================================================

# ── Stage 1: Dependencies ────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /install

COPY requirements.txt .

# Install lightweight CPU PyTorch wheels first
RUN pip install --no-cache-dir --prefix=/opt/venv torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining backend requirements
RUN pip install --no-cache-dir --prefix=/opt/venv -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install OpenCV system dependencies and ffmpeg
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built packages from the deps stage
COPY --from=deps /opt/venv /usr/local

# Copy application code and model weights
COPY app/ ./app/
COPY models/ ./models/

# Create evidence and models directories
RUN mkdir -p evidence models /tmp/evidence

# Create a non-root user for security (Hugging Face / Cloud compatibility)
RUN useradd -m -u 1000 trafficai && chown -R trafficai:trafficai /app /tmp/evidence
USER trafficai

# Expose standard port and HuggingFace Spaces port
EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, os; port=os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/api/health')"

# Start the server on $PORT or fallback to 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
