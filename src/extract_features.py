import parselmouth
import numpy as np
import librosa

def extract_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    
    # Pitch (mean)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_mean = np.mean(pitch_values[pitch_values > 0])
    
    # Jitter
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    
    # Shimmer
    shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # Harmonics-to-noise ratio (HNR)
    hnr = snd.to_harmonicity_cc().values
    hnr_mean = np.mean(hnr[hnr > 0])
    
    # MFCCs mean
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Combine features
    feature_vector = np.array([pitch_mean, jitter, shimmer, hnr_mean] + mfccs_mean.tolist())
    return feature_vector