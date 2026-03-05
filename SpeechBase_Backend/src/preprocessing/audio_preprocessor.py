import librosa
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import tempfile
import os

def preprocess_audio(file_path):
    """
    Preprocess audio file - handles multiple formats and converts to proper format
    """
    try:
        # If file_path is a file object (uploaded file), handle differently
        if hasattr(file_path, 'read'):
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(file_path.read())
                file_path.seek(0)  # Reset file pointer
                temp_path = tmp_file.name
            
            # Convert to WAV if needed
            audio_segment = AudioSegment.from_file(temp_path)
            audio_segment = audio_segment.set_channels(1)  # Convert to mono
            audio_segment = audio_segment.set_frame_rate(16000)  # Set sample rate
            
            # Export as WAV
            wav_path = temp_path.replace('.wav', '_converted.wav')
            audio_segment.export(wav_path, format='wav')
            
            # Load with librosa
            audio, sr = librosa.load(wav_path, sr=16000)
            
            # Cleanup
            os.unlink(temp_path)
            os.unlink(wav_path)
        else:
            # File path string
            # Try to convert format first using pydub
            try:
                audio_segment = AudioSegment.from_file(file_path)
                audio_segment = audio_segment.set_channels(1)  # Convert to mono
                audio_segment = audio_segment.set_frame_rate(16000)  # Set sample rate
                
                # Export to temporary WAV file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    temp_path = tmp_file.name
                
                audio_segment.export(temp_path, format='wav')
                
                # Load with librosa
                audio, sr = librosa.load(temp_path, sr=16000)
                
                # Cleanup
                os.unlink(temp_path)
            except:
                # If pydub fails, try direct librosa load
                audio, sr = librosa.load(file_path, sr=16000)
        
        # Noise reduction (basic normalization)
        audio = librosa.util.normalize(audio)
        
        return audio, sr
    
    except Exception as e:
        raise Exception(f"Error preprocessing audio: {str(e)}")