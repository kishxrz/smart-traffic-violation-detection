"""
backend/app/main.py
─────────────────────
FastAPI application entry point.

Responsibilities:
  - Configure logging.
  - Create the FastAPI app with metadata.
  - Configure CORS.
  - Register all routers.
  - Define lifespan (startup/shutdown events).
  - Provide a /api/config endpoint for the frontend.

Business logic lives in api/routes/, NOT in this file.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import analytics, detection, health, violations
from app.config import get_settings
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan handler.

    Startup: configure logging, create required directories, warm up model.
    Shutdown: release resources.
    """
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        json_format=settings.is_production,
    )

    logger.info("=" * 60)
    logger.info("  %s starting up", settings.app_name)
    logger.info("  Environment : %s", settings.app_env)
    logger.info("  Log level   : %s", settings.log_level)
    logger.info("  Device      : %s", settings.device)
    logger.info("=" * 60)

    # Create evidence directory
    settings.evidence_abs_dir.mkdir(parents=True, exist_ok=True)

    # Optionally warm up the YOLO model
    try:
        from app.api.dependencies import get_detector
        detector = get_detector()
        detector.warmup()
    except Exception as exc:
        logger.warning("Model warmup skipped: %s", exc)

    yield  # Application runs here

    logger.info("%s shutting down.", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Smart Traffic Intelligence & Violation Detection System. "
            "Detects vehicles, tracks movement, and identifies traffic violations "
            "using YOLOv8 + custom CV pipeline."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(detection.router)
    app.include_router(violations.router)
    app.include_router(analytics.router)

    # ── Config endpoint ───────────────────────────────────────────────────────
    @app.get("/api/config", tags=["config"])
    async def get_config():
        return {
            "app_name": settings.app_name,
            "environment": settings.app_env,
            "model_size": settings.yolo_model_size,
            "device": settings.device,
            "confidence_threshold": settings.confidence_threshold,
            "evidence_dir": str(settings.evidence_abs_dir),
        }

    return app


app = create_app()
