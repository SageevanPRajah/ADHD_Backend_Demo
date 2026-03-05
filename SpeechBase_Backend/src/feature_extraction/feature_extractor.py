import librosa
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import speech_recognition as sr

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Load SpaCy model (assuming English for now, need Sinhala model)
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading SpaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

def transcribe_audio(file_path):
    """
    Transcribe audio file to text using speech recognition
    Handles format conversion if needed
    """
    try:
        from pydub import AudioSegment
        import tempfile
        import os
        
        # Convert to WAV format if needed
        try:
            audio_segment = AudioSegment.from_file(file_path)
            audio_segment = audio_segment.set_channels(1)
            audio_segment = audio_segment.set_frame_rate(16000)
            
            # Export to temporary WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                temp_path = tmp_file.name
            
            audio_segment.export(temp_path, format='wav')
            actual_path = temp_path
        except:
            actual_path = file_path
        
        # Transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(actual_path) as source:
            audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data, language='si-LK')  # Sinhala
        except:
            try:
                text = recognizer.recognize_google(audio_data)  # Default to English
            except sr.UnknownValueError:
                text = ""
            except sr.RequestError:
                text = ""
        
        # Cleanup temp file
        if actual_path != file_path:
            try:
                os.unlink(actual_path)
            except:
                pass
        
        return text
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return ""

def extract_acoustic_features(audio, sr):
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Pitch
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    
    # Jitter, Shimmer (placeholder)
    jitter = 0.0
    shimmer = 0.0
    
    return {
        'mfccs': np.mean(mfccs, axis=1),
        'pitch': np.mean(pitches),
        'jitter': jitter,
        'shimmer': shimmer
    }

def extract_linguistic_features(text):
    if not text:
        return {}
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))  # Need Sinhala stopwords
    filtered_tokens = [w for w in tokens if not w in stop_words]
    
    # Lexical diversity
    lexical_diversity = len(set(filtered_tokens)) / len(filtered_tokens) if filtered_tokens else 0
    
    # Narrative coherence (placeholder)
    coherence = 0.5
    
    # Filler words (placeholder)
    fillers = 0
    
    return {
        'lexical_diversity': lexical_diversity,
        'coherence': coherence,
        'fillers': fillers
    }

def extract_features(audio, sr, text=None):
    acoustic = extract_acoustic_features(audio, sr)
    if text is None or text == "":
        linguistic = {
            'lexical_diversity': 0.0,
            'coherence': 0.0,
            'fillers': 0
        }
    else:
        linguistic = extract_linguistic_features(text)
    features = {**acoustic, **linguistic}
    # Flatten mfccs
    mfccs_flat = features.pop('mfccs')
    for i, val in enumerate(mfccs_flat):
        features[f'mfcc_{i}'] = val
    return features