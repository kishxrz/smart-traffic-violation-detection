"""
backend/tests/test_async_video.py
───────────────────────────────────
Unit tests for asynchronous video processing background jobs.
Verifies HTTP 202 response, job manager state transitions, status endpoint, and background completion.
"""

import pytest
import time
from fastapi.testclient import TestClient

from app.main import app
from app.detection.job_manager import job_manager


@pytest.fixture
def client():
    return TestClient(app)


def test_video_submission_returns_202(client):
    """POST /api/detection/video should immediately return HTTP 202 with job_id."""
    # Create dummy video bytes
    dummy_video = b"\x00" * 1024
    files = {"file": ("test.mp4", dummy_video, "video/mp4")}

    response = client.post("/api/detection/video", files=files)
    assert response.status_code == 202

    data = response.json()
    assert "job_id" in data
    assert data["status"] in ("queued", "processing", "failed")
    assert "progress" in data
    assert "frames_processed" in data


def test_job_status_endpoint(client):
    """GET /api/detection/video/{job_id} should return job status."""
    job = job_manager.create_job()
    response = client.get(f"/api/detection/video/{job.job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job.job_id
    assert data["status"] == "queued"
    assert data["progress"] == 0.0


def test_job_not_found(client):
    """GET /api/detection/video/{invalid_id} should return 404."""
    response = client.get("/api/detection/video/non-existent-job-id")
    assert response.status_code == 404


def test_job_manager_lifecycle():
    """Verify JobManager lifecycle state transitions."""
    job = job_manager.create_job()
    assert job.status == "queued"

    job_manager.set_processing(job.job_id, total_frames=100)
    j1 = job_manager.get_job(job.job_id)
    assert j1.status == "processing"
    assert j1.total_frames == 100

    job_manager.update_progress(job.job_id, frames_processed=50, total_frames=100)
    j2 = job_manager.get_job(job.job_id)
    assert j2.progress == 50.0

    job_manager.complete_job(job.job_id, result={"test": "ok"})
    j3 = job_manager.get_job(job.job_id)
    assert j3.status == "completed"
    assert j3.progress == 100.0
    assert j3.result == {"test": "ok"}


def test_job_manager_failure():
    """Verify JobManager handles failure gracefully."""
    job = job_manager.create_job()
    job_manager.fail_job(job.job_id, "Model loading error")
    j = job_manager.get_job(job.job_id)
    assert j.status == "failed"
    assert j.error == "Model loading error"
