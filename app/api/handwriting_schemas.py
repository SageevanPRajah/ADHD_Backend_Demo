"""Pydantic schemas for the handwriting ADHD prediction endpoint."""

from pydantic import BaseModel, Field
from typing import List, Optional


class StrokePoint(BaseModel):
    type: str = Field(..., description="'start' or 'move'")
    x: float
    y: float
    timestamp: float


class HandwritingSessionIn(BaseModel):
    """Matches the JSON structure sent by the frontend HandwritingGame canvas."""
    grade: Optional[str] = None
    activity: Optional[str] = None
    instruction: Optional[str] = None
    penSize: Optional[float] = 8.0
    timestamp: Optional[str] = None
    strokes: List[StrokePoint]


class HandwritingPredictionOut(BaseModel):
    prediction: str = Field(..., description="'ADHD Risk' or 'No ADHD Risk'")
    probability: float = Field(..., description="ADHD probability 0-1")
    risk_level: str = Field(..., description="'High', 'Moderate', or 'Low'")
    adhd_probability: float
    risk_score: str = Field(..., description="Human-readable percentage e.g. '67%'")
