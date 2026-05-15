import streamlit as st
import joblib
import tempfile
import os
import numpy as np
# The component itself, correctly installed
from st_audiorec import st_audiorec 

# --- Streamlit Configuration ---
st.set_page_config(page_title="🎙️ Voice Analysis for Parkinson's", layout="centered")

# --- Dependency Check and Mock Function ---
try:
    # Ensure this import is correct:
    from src.audio_prepro import extract_features
except ImportError:
    st.error("🚨 Error: Could not import 'extract_features' from 'src.audio_prepro'.")
    st.info("Please ensure you have a directory named 'src' with an 'audio_prepro.py' file inside.")
    
    def extract_features(file_path):
        st.warning("Using Mock Feature Extraction: Please fix the 'src' import!")
        # Assumes model expects 21 features (like the UCI dataset)
        return np.array([0.5] * 21)


# --- 1. Model Loading (Cached) ---

@st.cache_resource
def load_parkinsons_model():
    """Loads the trained Random Forest model."""
    try:
        # NOTE: Ensure 'parkinsons_rf_model.pkl' is in the directory you run app.py from.
        model = joblib.load('parkinsons_rf_model.pkl')
        st.success("✅ ML Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("Model file 'parkinsons_rf_model.pkl' not found.")
        st.stop()

model = load_parkinsons_model()


# --- 2. Prediction Function ---

def predict_on_audio(audio_bytes):
    """
    Saves raw WAV audio bytes to a temp file, extracts features, predicts, and cleans up.
    """
    final_wav_path = None
    
    try:
        # 1. Save the raw WAV bytes directly to a temporary file
        final_wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with open(final_wav_path, 'wb') as f:
            f.write(audio_bytes)
        
        # 2. Preprocess and Extract Features
        # The file is a clean WAV, so extract_features should work seamlessly.
        features = extract_features(final_wav_path).reshape(1, -1)
        
        # 3. Predict
        risk = model.predict(features)[0]
        confidence = model.predict_proba(features).max()
        
        return {'label': int(risk), 'confidence': float(confidence)}
    
    except Exception as e:
        # This should only catch errors in feature extraction or model prediction now.
        st.error(f"Prediction Error: Could not process audio or predict. Details: {e}")
        return None
    
    finally:
        # 4. Cleanup
        if final_wav_path and os.path.exists(final_wav_path):
            os.remove(final_wav_path)

# --- 3. Streamlit UI and Execution ---

st.title("🎙️ Voice Analysis for Parkinson's Risk")
st.markdown("Record a short 'ah' sound. The audio will be analyzed by a Random Forest model to assess risk.")


# --- COMPONENT CALL (FINAL & BARE FIX) ---
# Call st_audiorec with NO arguments to avoid the TypeError
audio_bytes = st_audiorec()
# ------------------------------------------


# Only run prediction when the component has returned data (which is raw WAV bytes)
if audio_bytes is not None:
    st.info("✅ Audio recording complete. Sending to model...")

    with st.spinner('Running AI Model...'):
        # Pass raw bytes directly to the prediction function
        result = predict_on_audio(audio_bytes)
    
    # --- Display Result ---
    if result:
        st.subheader("📊 Prediction Results")
        label = result['label']
        confidence = result['confidence']
        
        status_text = 'Parkinson’s Risk' if label == 1 else 'Healthy'
        
        if label == 1:
            st.error(f"**🔴 Prediction: {status_text}**")
            st.metric("Confidence", f"{(confidence * 100):.2f}%")
            st.warning("⚠️ **Disclaimer:** This is a screening tool, not a medical diagnosis. Consult a professional.")
        else:
            st.success(f"**🟢 Prediction: {status_text}**")
            st.metric("Confidence", f"{(confidence * 100):.2f}%")
            st.balloons()
            
        # Optional: Playback the recorded audio
        st.markdown("---")
        st.caption("Review your recording:")
        st.audio(audio_bytes, format='audio/wav')