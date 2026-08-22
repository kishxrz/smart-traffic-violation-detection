"""
backend/app/api/routes/violations.py
──────────────────────────────────────
Violation endpoints.

GET /api/violations         — List all violations for current session.
GET /api/violations/{id}    — Get a specific violation by ID.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_analytics, get_violation_engine
from app.analytics.metrics import AnalyticsAccumulator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/violations", tags=["violations"])

# In-memory store for the current session's violations.
# Production: replace with PostgreSQL queries.
_session_violations: List[dict] = []


def store_violations(violations: list) -> None:
    """Called by the processing pipeline to store violations."""
    global _session_violations
    for v in violations:
        _session_violations.append(v.model_dump())


def clear_violations() -> None:
    global _session_violations
    _session_violations = []


@router.get("")
async def list_violations(
    limit: int = 100,
    offset: int = 0,
    violation_type: str = None,
) -> Dict[str, Any]:
    """
    Return violations for the current session.

    Query params:
      limit:          Max number of records to return.
      offset:         Pagination offset.
      violation_type: Filter by type (e.g., "NO_HELMET").
    """
    violations = _session_violations

    if violation_type:
        violations = [
            v for v in violations
            if v.get("violation_type") == violation_type.upper()
        ]

    total = len(violations)
    page = violations[offset : offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "violations": page,
    }


@router.get("/{violation_id}")
async def get_violation(violation_id: str) -> Dict[str, Any]:
    """Get a specific violation by its UUID."""
    for v in _session_violations:
        if v.get("id") == violation_id:
            return v
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Violation {violation_id!r} not found.",
    )
