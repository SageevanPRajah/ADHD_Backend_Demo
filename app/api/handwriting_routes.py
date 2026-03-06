"""API route for handwriting-based ADHD prediction."""

from fastapi import APIRouter, HTTPException

from app.api.handwriting_schemas import HandwritingSessionIn, HandwritingPredictionOut
from app.services.handwriting_service import predict_adhd

router = APIRouter(prefix="/handwriting", tags=["Handwriting"])


@router.post("/predict", response_model=HandwritingPredictionOut)
async def predict_handwriting(session: HandwritingSessionIn):
    """
    Receive drawing-session stroke data from the frontend canvas,
    extract behavioural features, and return an ADHD risk score.
    """
    if not session.strokes or len(session.strokes) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 stroke data points are required for prediction.",
        )

    try:
        stroke_dicts = [pt.model_dump() for pt in session.strokes]
        result = predict_adhd(stroke_dicts, pen_size=session.penSize or 8.0)
        return HandwritingPredictionOut(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
