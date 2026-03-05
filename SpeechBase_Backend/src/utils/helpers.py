# Utility functions
def load_audio(file_path):
    import librosa
    return librosa.load(file_path)

def save_model(model, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f)