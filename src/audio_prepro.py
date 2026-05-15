import librosa
import numpy as np
import parselmouth

# --- Preprocessing Function (Included for completeness) ---

def preprocess_audio(audio_path, sr_target=44100):
    audio, sr = librosa.load(audio_path, sr=None)  # load audio with original sr
    if sr != sr_target:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_target)  # resample
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    # Trim silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    return audio_trimmed, sr_target

# --- Feature Extraction Function (FIXED HNR CALL) ---

def extract_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    
    # 1. ParSelMouth Features (Jitter, Shimmer, Pitch, HNR)
    
    # Pitch object for fundamental frequency measures
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    
    # Filter out unvoiced frames (pitch = 0)
    voiced_pitch_values = pitch_values[pitch_values > 0]
    
    # Handle the edge case of all unvoiced audio
    if len(voiced_pitch_values) == 0:
        # If no voice is detected, return zeros for pitch-based features
        pitch_mean, pitch_max, pitch_min = 0.0, 0.0, 0.0
    else:
        # 1. Mean Pitch (1 feature)
        pitch_mean = np.mean(voiced_pitch_values)
        # 2. Maximum Pitch (1 feature)
        pitch_max = np.max(voiced_pitch_values)
        # 3. Minimum Pitch (1 feature)
        pitch_min = np.min(voiced_pitch_values)

    # 4. Jitter (local) (1 feature)
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    
    # 5. Shimmer (local) (1 feature)
    shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # 6. Harmonics-to-noise ratio (HNR) (1 feature)
    # FIX: Create Harmonicity object first, then ask for its mean.
    harmonicity = snd.to_harmonicity() 
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    
    # 2. Librosa Features (MFCCs and Spectral properties)
    
    # Reload audio for Librosa spectral features
    y, sr = librosa.load(audio_path)
    
    # 7-19. MFCCs mean (13 features)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # 20. Spectral Centroid Mean (1 feature)
    centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # 21. Spectral Bandwidth Mean (1 feature)
    bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    
    # 22. Zero Crossing Rate Mean (1 feature)
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    # --- Combine all 22 features ---
    feature_vector = np.hstack([
        [pitch_mean, pitch_max, pitch_min, jitter, shimmer, hnr], # 6 features
        mfccs_mean,                                              # 13 features
        [centroid_mean, bandwidth_mean, zcr_mean]                # 3 features
    ])

    return feature_vector