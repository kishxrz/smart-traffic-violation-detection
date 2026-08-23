"""
backend/app/detection/job_manager.py
───────────────────────────────────
File-backed job manager for tracking asynchronous video processing jobs.
Thread-safe and process-safe across Uvicorn workers and background processes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Any

from app.config import get_settings
from app.schemas.job import VideoJobResponse, JobStatus

logger = logging.getLogger(__name__)


class JobManager:
    """Process-safe, file-backed manager for background video processing jobs."""

    def __init__(self, max_history: int = 100):
        self._lock = threading.Lock()
        self._max_history = max_history
        self._file_path = get_settings().evidence_abs_dir / "jobs.json"

    def _load_jobs(self) -> Dict[str, VideoJobResponse]:
        """Load jobs from persistent json file."""
        if not self._file_path.exists():
            return {}
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: VideoJobResponse(**v) for k, v in data.items()}
        except Exception as err:
            logger.warning("Could not read jobs file %s: %s", self._file_path, err)
            return {}

    def _save_jobs(self, jobs: Dict[str, VideoJobResponse]) -> None:
        """Write jobs dictionary atomically to persistent json file."""
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.model_dump() for k, v in jobs.items()}
            temp_file = self._file_path.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self._file_path)
        except Exception as err:
            logger.error("Could not write jobs file %s: %s", self._file_path, err)

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
            jobs = self._load_jobs()
            if len(jobs) >= self._max_history:
                finished_keys = [k for k, v in jobs.items() if v.status in ("completed", "failed")]
                for k in finished_keys[: len(jobs) - self._max_history + 1]:
                    del jobs[k]

            jobs[job_id] = job
            self._save_jobs(jobs)

        logger.info("Created background video job: %s", job_id)
        return job

    def get_job(self, job_id: str) -> Optional[VideoJobResponse]:
        """Retrieve job details by job_id."""
        with self._lock:
            jobs = self._load_jobs()
            return jobs.get(job_id)

    def set_processing(self, job_id: str, total_frames: Optional[int] = None) -> None:
        """Mark job as processing."""
        with self._lock:
            jobs = self._load_jobs()
            job = jobs.get(job_id)
            if job:
                job.status = "processing"
                if total_frames and total_frames > 0:
                    job.total_frames = total_frames
                jobs[job_id] = job
                self._save_jobs(jobs)

    def update_progress(self, job_id: str, frames_processed: int, total_frames: Optional[int] = None) -> None:
        """Update frames processed and calculate progress percentage."""
        with self._lock:
            jobs = self._load_jobs()
            job = jobs.get(job_id)
            if not job:
                return

            job.status = "processing"
            job.frames_processed = frames_processed
            if total_frames and total_frames > 0:
                job.total_frames = total_frames

            if job.total_frames and job.total_frames > 0:
                job.progress = round(min(100.0, (frames_processed / job.total_frames) * 100.0), 1)

            jobs[job_id] = job
            self._save_jobs(jobs)

    def complete_job(self, job_id: str, result: Dict[str, Any]) -> None:
        """Mark job as successfully completed with final result dictionary."""
        with self._lock:
            jobs = self._load_jobs()
            job = jobs.get(job_id)
            if job:
                job.status = "completed"
                job.progress = 100.0
                job.result = result
                job.completed_at = time.time()
                jobs[job_id] = job
                self._save_jobs(jobs)
                logger.info("Job %s completed successfully", job_id)

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed with error details."""
        with self._lock:
            jobs = self._load_jobs()
            job = jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error = error
                job.completed_at = time.time()
                jobs[job_id] = job
                self._save_jobs(jobs)
                logger.error("Job %s failed: %s", job_id, error)


# Global singleton instance
job_manager = JobManager()
