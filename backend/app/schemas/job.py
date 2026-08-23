"""
backend/app/schemas/job.py
─────────────────────────
Pydantic schemas for asynchronous video processing jobs.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field
import uuid
import time

JobStatus = Literal["queued", "processing", "completed", "failed"]


class VideoJobResponse(BaseModel):
    """API response model for video processing job status and results."""
    job_id: str
    status: JobStatus = "queued"
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    frames_processed: int = 0
    total_frames: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
