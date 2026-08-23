"""
backend/app/config.py
─────────────────────
Typed configuration layer for the entire backend.

All settings are read from environment variables (or a .env file).
This single config object is imported everywhere — no scattered
os.getenv() calls in business logic.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root calculation (handles both local repo and Docker container structure)
_parent1 = Path(__file__).resolve().parents[1]
_parent2 = Path(__file__).resolve().parents[2]
if (_parent1 / "models").exists():
    PROJECT_ROOT = _parent1
else:
    PROJECT_ROOT = _parent2


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "Smart Traffic AI"
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = False
    app_port: int = 8000
    app_host: str = "0.0.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./traffic.db"
    database_echo: bool = False

    # ── Model Paths ───────────────────────────────────────────────────────────
    yolo_model_path: str = "models/yolov8n.pt"
    yolo_model_size: Literal["n", "s", "m", "l", "x"] = "n"
    helmet_model_path: Optional[str] = "models/helmet_v3.pt"

    # ── Inference Hyperparameters ─────────────────────────────────────────────
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    helmet_confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    inference_fps: int = Field(default=10, ge=1, le=120)
    frame_skip: int = Field(default=2, ge=1, le=60)
    device: str = "cpu"  # "cpu" | "cuda" | "mps"

    # ── Evidence ─────────────────────────────────────────────────────────────
    evidence_dir: str = "evidence"
    evidence_cooldown_seconds: float = 3.0

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # ── Upload Limits ─────────────────────────────────────────────────────────
    max_upload_size_mb: int = 500

    # ─────────────────────────────────────────────────────────────────────────

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list) -> list:
        """Accept JSON string or plain list."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    @property
    def yolo_model_abs_path(self) -> Path:
        """Resolve YOLO model path relative to project root."""
        p = Path(self.yolo_model_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def helmet_model_abs_path(self) -> Optional[Path]:
        if not self.helmet_model_path:
            return None
        p = Path(self.helmet_model_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def evidence_abs_dir(self) -> Path:
        p = Path(self.evidence_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton of Settings.

    Using @lru_cache means the .env file is read exactly once at startup.
    Tests can clear the cache with get_settings.cache_clear().
    """
    return Settings()
