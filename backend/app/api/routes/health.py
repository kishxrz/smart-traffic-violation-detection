"""
backend/app/api/routes/health.py
──────────────────────────────────
Health check endpoint.

GET /api/health
  Returns system status, model info, and version.
  Used by Docker healthcheck, load balancers, and monitoring.
"""

from __future__ import annotations

import platform
import sys
import time
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter(prefix="/api/health", tags=["health"])

_start_time = time.time()


@router.get("")
async def health_check() -> Dict[str, Any]:
    """
    System health check.

    Returns:
      - status: "ok" if the service is running.
      - uptime_seconds: time since application start.
      - python_version: Python runtime version.
      - platform: OS info.
    """
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "python_version": sys.version,
        "platform": platform.platform(),
        "service": "Smart Traffic AI Backend",
        "version": "1.0.0",
    }
