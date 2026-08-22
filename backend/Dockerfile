# ============================================================
# Smart Traffic AI — Backend Dockerfile
# Multi-stage build for a lean production image.
# ============================================================

# ── Stage 1: Dependencies ────────────────────────────────────
FROM python:3.13-slim AS deps

WORKDIR /install

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/opt/venv -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built packages from the deps stage
COPY --from=deps /opt/venv /usr/local

# Copy application code
COPY app/ ./app/

# Create evidence and models directories
RUN mkdir -p evidence models

# Create a non-root user for security
RUN useradd -m -u 1001 trafficai && chown -R trafficai:trafficai /app
USER trafficai

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
