"""
backend/tests/test_api.py
──────────────────────────
Integration tests for FastAPI endpoints using httpx async client.

Tests the health, detection/config, analytics, and violations endpoints
without requiring a real video file or GPU.
"""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac


@pytest.mark.anyio
async def test_health_check(client: AsyncClient):
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "python_version" in data


@pytest.mark.anyio
async def test_detection_config(client: AsyncClient):
    response = await client.get("/api/detection/config")
    assert response.status_code == 200
    data = response.json()
    assert "confidence_threshold" in data
    assert "iou_threshold" in data
    assert "device" in data


@pytest.mark.anyio
async def test_analytics_summary(client: AsyncClient):
    response = await client.get("/api/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_frames_processed" in data
    assert "total_violations" in data


@pytest.mark.anyio
async def test_violations_list(client: AsyncClient):
    response = await client.get("/api/violations")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "violations" in data
    assert isinstance(data["violations"], list)


@pytest.mark.anyio
async def test_violation_not_found(client: AsyncClient):
    response = await client.get("/api/violations/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_api_config(client: AsyncClient):
    response = await client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "app_name" in data
    assert "environment" in data
