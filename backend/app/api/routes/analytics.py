"""
backend/app/api/routes/analytics.py
─────────────────────────────────────
Analytics endpoints.

GET /api/analytics/summary    — Aggregated session statistics.
GET /api/analytics/violations — Violation breakdown by type and severity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.analytics.metrics import AnalyticsAccumulator
from app.api.dependencies import get_analytics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary")
async def analytics_summary(
    analytics: AnalyticsAccumulator = Depends(get_analytics),
) -> Dict[str, Any]:
    """
    Return the aggregated analytics summary for the current session.

    Includes:
      - Total vehicles detected and unique vehicles tracked.
      - Violation counts by type and severity.
      - Average inference time and processing FPS.
      - Vehicle class distribution.
      - Traffic volume time series.
    """
    summary = analytics.build_summary()
    return summary.model_dump()


@router.get("/violations")
async def violation_analytics(
    analytics: AnalyticsAccumulator = Depends(get_analytics),
) -> Dict[str, Any]:
    """Return violation breakdown suitable for chart rendering."""
    summary = analytics.build_summary()
    return {
        "total_violations": summary.total_violations,
        "by_type": [
            {"type": k, "count": v}
            for k, v in summary.violations_by_type.items()
        ],
        "by_severity": [
            {"severity": k, "count": v}
            for k, v in summary.violations_by_severity.items()
        ],
        "violations_per_minute": round(summary.violations_per_minute, 2),
    }
