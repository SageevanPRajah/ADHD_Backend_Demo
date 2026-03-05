# ADHD Speech-Based Detection API

A FastAPI-based REST API for AI-driven speech screening to detect Attention Deficit Hyperactivity Disorder (ADHD) in Sinhala-speaking children.

## Features

- 🔌 **REST API** - FastAPI-based endpoints for audio analysis
- 📁 **File Upload** - Accept audio files (WAV, MP3, M4A, FLAC, OGG)
- 🔊 **Audio Processing** - Automatic format conversion and noise reduction
- 🧠 **Feature Extraction** - Acoustic (MFCCs, pitch, jitter, shimmer) and linguistic features
- 📊 **ADHD Classification** - ML-based screening with probability scores
- 📝 **Speech Transcription** - Automatic transcription using speech recognition
- 💡 **Recommendations** - Actionable insights based on screening results

## Prerequisites

### System Requirements
- Python 3.10 or higher
- ffmpeg (audio processing)

### Installing ffmpeg

**Windows:**
1. Download from https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to System PATH

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

## Installation

1. **Clone or extract the project**
```bash
cd path/to/jigan
```

2. **Create a virtual environment (recommended)**
```bash
# Windows
python -m venv jigan
jigan\Scripts\activate

# Linux/macOS
python3 -m venv jigan
source jigan/bin/activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Download SpaCy language model**
```bash
python -m spacy download en_core_web_sm
```

5. **Download NLTK data**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

## Running the Application

### Backend (FastAPI)

1. **Start the API server**
```bash
# Using uvicorn directly
uvicorn main:app --reload

# Or run the main.py file
python main.py
```

**API Documentation:** `http://localhost:8000/docs`

### Frontend (React)

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start the development server**
```bash
npm start
```

**Frontend URL:** `http://localhost:3000`

### Quick Start (Both Services)

For convenience, you can start both backend and frontend with a single command:

```bash
python start.py
```

This will start both servers and provide access URLs.

### Manual Startup

**Terminal 1 (Backend):**
```bash
uvicorn main:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend && npm start
```

The complete application will be available at:
- **Frontend:** `http://localhost:3000`
- **API:** `http://localhost:8000`
- **API Docs:** `http://localhost:8000/docs`

## API Endpoints

### POST /analyze
Analyze audio file for ADHD indicators.

**Parameters:**
- `file` (required): Audio file upload (WAV, MP3, M4A, FLAC, OGG)
- `child_age` (optional): Child's age (6-12, default: 8)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav" \
     -F "child_age=8"
```

**Response:**
```json
{
  "child_age": 8,
  "analysis": {
    "probability": 0.7234,
    "classification": "ADHD Indicators Detected",
    "confidence": "High",
    "transcription_available": true
  },
  "features": {
    "lexical_diversity": 0.4567,
    "coherence": 0.5432,
    "fillers": 3,
    "pitch_mean": 234.5678,
    "jitter": 0.0123,
    "shimmer": 0.0345
  },
  "transcription": "මම ගෙදර බල්ලෙක් හදනවා...",
  "recommendations": {
    "warning": "This screening suggests potential ADHD indicators.",
    "note": "This is a screening tool, not a diagnostic test...",
    "advice": "Early intervention can significantly improve outcomes."
  }
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running"
}
```

### GET /
API information endpoint.

## Project Structure

```
jigan/
├── main.py                        # Main FastAPI application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── frontend/                       # React frontend application
│   ├── src/
│   │   ├── components/            # React components
│   │   ├── App.js                 # Main React app
│   │   └── index.js               # React entry point
│   ├── public/                    # Static assets
│   ├── package.json               # Frontend dependencies
│   └── README.md                  # Frontend documentation
├── data/                          # Data storage
│   └── collect_data.py           # Data collection utilities
├── models/                        # Trained ML models
│   └── adhd_model.pkl            # Pre-trained classifier (if available)
├── src/
│   ├── preprocessing/
│   │   └── audio_preprocessor.py # Audio preprocessing functions
│   ├── feature_extraction/
│   │   └── feature_extractor.py  # Feature extraction (acoustic + linguistic)
│   ├── models/
│   │   ├── classifier.py         # ADHD classification model
│   │   └── train_model.py        # Model training script
│   └── utils/
│       └── helpers.py            # Utility functions
├── notebooks/                     # Jupyter notebooks for experiments
└── tests/                         # Unit tests
    └── test_features.py
```

## Features Extracted

### Acoustic Features
- **MFCCs** (13 coefficients) - Voice quality characteristics
- **Pitch** - Fundamental frequency
- **Jitter** - Voice stability
- **Shimmer** - Amplitude variation

### Linguistic Features
- **Lexical Diversity** - Vocabulary richness
- **Narrative Coherence** - Story structure quality
- **Filler Words** - Use of "um", "uh", etc.
- **Speech Fluency** - Hesitation patterns

## Model Training

To train a new model with your own data:

1. Collect labeled speech samples
2. Extract features using `feature_extractor.py`
3. Run training script:
```bash
python src/models/train_model.py
```

## Important Notes

⚠️ **This is a screening tool, not a diagnostic instrument**
- Results should be interpreted by healthcare professionals
- Not a substitute for clinical diagnosis
- Use as part of comprehensive ADHD assessment

## Troubleshooting

**API not starting:**
- Ensure port 8000 is not in use
- Check for import errors: `python -c "import fastapi; print('FastAPI OK')"`

**Audio processing errors:**
- Verify ffmpeg is installed and in PATH
- Check file format is supported (WAV, MP3, M4A, FLAC, OGG)

**Import errors:**
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

**SpaCy model not found:**
- Run: `python -m spacy download en_core_web_sm`

**Large file uploads failing:**
- Check server upload limits
- Ensure sufficient disk space for temporary files

## API Testing

Test the API with sample requests:

```bash
# Health check
curl http://localhost:8000/health

# API info
curl http://localhost:8000/

# Analyze audio (replace with actual file)
curl -X POST "http://localhost:8000/analyze" \
     -F "file=@sample_audio.wav" \
     -F "child_age=8"
```

## Development

To run in development mode with auto-reload:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Contributing

To contribute to this project:
1. Report issues via issue tracker
2. Submit pull requests for improvements
3. Suggest new features or enhancements

## License

This project is for research and educational purposes.

## Support

For questions or issues, please contact the development team.

---

**Developed as part of ADHD detection research for Sinhala-speaking children in Sri Lanka**