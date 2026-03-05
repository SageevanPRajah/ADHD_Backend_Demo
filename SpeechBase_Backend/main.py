from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import uvicorn
from typing import Dict, Any
import traceback

# Import our modules
from src.preprocessing.audio_preprocessor import preprocess_audio
from src.feature_extraction.feature_extractor import extract_features, transcribe_audio
from src.models.classifier import classify_adhd

app = FastAPI(
    title="ADHD Speech Detection API",
    description="API for analyzing speech patterns to screen for ADHD indicators in Sinhala-speaking children",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3002", "http://127.0.0.1:3002"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ADHD Speech Detection API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /analyze": "Analyze audio for ADHD indicators"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    child_age: int = 8
) -> Dict[str, Any]:
    """
    Analyze uploaded audio file for ADHD speech indicators

    Args:
        file: Audio file (WAV, MP3, etc.)
        child_age: Child's age (6-12)

    Returns:
        Analysis results including probability, classification, and features
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file type
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm']
    file_extension = os.path.splitext(file.filename.lower())[1]
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Validate age
    if not (6 <= child_age <= 12):
        raise HTTPException(status_code=400, detail="Child age must be between 6 and 12")

    temp_file_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        # Preprocess audio
        audio, sr = preprocess_audio(temp_file_path)

        # Transcribe audio
        transcription = transcribe_audio(temp_file_path)

        # Extract features
        features = extract_features(audio, sr, transcription)

        # Classify ADHD
        result = classify_adhd(features)

        # Calculate confidence
        confidence = "High" if abs(result['probability'] - 0.5) > 0.3 else "Medium"

        # Prepare response
        response = {
            "child_age": child_age,
            "analysis": {
                "probability": round(float(result['probability']), 4),
                "classification": "ADHD Indicators Detected" if result['adhd'] else "No ADHD Indicators",
                "confidence": confidence,
                "transcription_available": bool(transcription.strip())
            },
            "features": {
                "lexical_diversity": round(float(features.get('lexical_diversity', 0)), 4),
                "coherence": round(float(features.get('coherence', 0)), 4),
                "fillers": int(features.get('fillers', 0)),
                "pitch_mean": round(float(features.get('pitch', 0)), 4),
                "jitter": round(float(features.get('jitter', 0)), 4),
                "shimmer": round(float(features.get('shimmer', 0)), 4)
            }
        }

        # Add transcription if available
        if transcription.strip():
            response["transcription"] = transcription

        # Add recommendations
        if result['adhd']:
            response["recommendations"] = {
                "warning": "This screening suggests potential ADHD indicators.",
                "note": "This is a screening tool, not a diagnostic test. Consult a healthcare professional for proper evaluation.",
                "advice": "Early intervention can significantly improve outcomes."
            }
        else:
            response["recommendations"] = {
                "message": "No significant ADHD indicators detected in this screening.",
                "advice": "Continue monitoring speech and behavior patterns. Regular developmental check-ups are recommended."
            }

        return response

    except Exception as e:
        error_detail = str(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_detail}")

    finally:
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)