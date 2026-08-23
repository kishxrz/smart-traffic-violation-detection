"""
backend/app/detection/job_manager.py
───────────────────────────────────
In-memory job manager for asynchronous video processing jobs.
Thread-safe for concurrency across API endpoints and background workers.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict, Optional, Any

from app.schemas.job import VideoJobResponse, JobStatus

logger = logging.getLogger(__name__)


class JobManager:
    """Thread-safe in-memory manager for tracking background video processing jobs."""

    def __init__(self, max_history: int = 100):
        self._jobs: Dict[str, VideoJobResponse] = {}
        self._lock = threading.Lock()
        self._max_history = max_history

    def create_job(self) -> VideoJobResponse:
        """Create and store a new queued video job."""
        job_id = str(uuid.uuid4())
        job = VideoJobResponse(
            job_id=job_id,
            status="queued",
            progress=0.0,
            frames_processed=0,
            created_at=time.time(),
        )
        with self._lock:
            # Clean up old completed/failed jobs if exceeding history limit
            if len(self._jobs) >= self._max_history:
                finished_keys = [k for k, v in self._jobs.items() if v.status in ("completed", "failed")]
                for k in finished_keys[: len(self._jobs) - self._max_history + 1]:
                    del self._jobs[k]

            self._jobs[job_id] = job

        logger.info("Created background video job: %s", job_id)
        return job

    def get_job(self, job_id: str) -> Optional[VideoJobResponse]:
        """Retrieve job details by job_id."""
        with self._lock:
            return self._jobs.get(job_id)

    def set_processing(self, job_id: str, total_frames: Optional[int] = None) -> None:
        """Mark job as processing."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "processing"
                if total_frames and total_frames > 0:
                    job.total_frames = total_frames

    def update_progress(self, job_id: str, frames_processed: int, total_frames: Optional[int] = None) -> None:
        """Update frames processed and calculate progress percentage."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            job.status = "processing"
            job.frames_processed = frames_processed
            if total_frames and total_frames > 0:
                job.total_frames = total_frames

            if job.total_frames and job.total_frames > 0:
                job.progress = round(min(100.0, (frames_processed / job.total_frames) * 100.0), 1)

    def complete_job(self, job_id: str, result: Dict[str, Any]) -> None:
        """Mark job as successfully completed with final result dictionary."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "completed"
                job.progress = 100.0
                job.result = result
                job.completed_at = time.time()
                logger.info("Job %s completed successfully", job_id)

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed with error details."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error = error
                job.completed_at = time.time()
                logger.error("Job %s failed: %s", job_id, error)


# Global singleton instance
job_manager = JobManager()
