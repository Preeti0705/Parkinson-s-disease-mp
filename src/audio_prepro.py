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

# --- Feature Extraction Function for Telemonitoring Dataset ---

def extract_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    
    # 1. Pitch & PointProcess
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
    
    # 2. Jitter features
    jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_abs = parselmouth.praat.call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp = parselmouth.praat.call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    
    # 3. Shimmer features
    shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db = parselmouth.praat.call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda = parselmouth.praat.call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # 4. HNR
    harmonicity = snd.to_harmonicity() 
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    
    # Handle undefined values (NaNs)
    import math
    def clean_val(val):
        if val is None or math.isnan(val):
            return 0.0
        return val

    jitter_local = clean_val(jitter_local)
    jitter_abs = clean_val(jitter_abs)
    jitter_rap = clean_val(jitter_rap)
    jitter_ppq5 = clean_val(jitter_ppq5)
    jitter_ddp = clean_val(jitter_ddp)
    
    shimmer_local = clean_val(shimmer_local)
    shimmer_db = clean_val(shimmer_db)
    shimmer_apq3 = clean_val(shimmer_apq3)
    shimmer_apq5 = clean_val(shimmer_apq5)
    shimmer_apq11 = clean_val(shimmer_apq11)
    shimmer_dda = clean_val(shimmer_dda)
    
    hnr = clean_val(hnr)
    # NHR = 10^(-HNR / 10)
    nhr = 10 ** (-hnr / 10) if hnr != 0 else 0.0
    
    return np.array([
        jitter_local, jitter_abs, jitter_rap, jitter_ppq5, jitter_ddp,
        shimmer_local, shimmer_db, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda,
        nhr, hnr
    ])