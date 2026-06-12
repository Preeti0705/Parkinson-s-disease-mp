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
        # Assumes model expects 13 acoustic features
        return np.array([0.5] * 13)


# --- 1. Model and Scaler Loading (Cached) ---

@st.cache_resource
def load_parkinsons_model_and_scaler():
    """Loads the trained Random Forest regressor and scaler."""
    try:
        model = joblib.load('parkinsons_rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        st.success("✅ ML Model and Scaler loaded successfully!")
        return model, scaler
    except FileNotFoundError:
        st.error("Model file 'parkinsons_rf_model.pkl' or 'scaler.pkl' not found. Please run 'python src/model.py' to train the model first.")
        st.stop()

model, scaler = load_parkinsons_model_and_scaler()


# --- 2. Prediction Function ---

def predict_on_audio(audio_bytes, age, sex):
    """
    Saves raw WAV audio bytes to a temp file, extracts features, predicts UPDRS score, and cleans up.
    """
    final_wav_path = None
    
    try:
        # 1. Save the raw WAV bytes directly to a temporary file
        final_wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with open(final_wav_path, 'wb') as f:
            f.write(audio_bytes)
        
        # 2. Preprocess and Extract Acoustic Features
        acoustic_features = extract_features(final_wav_path)
        
        # 3. Combine with demographic features: age, sex (0 for Male, 1 for Female)
        full_features = np.hstack([[age, sex], acoustic_features]).reshape(1, -1)
        
        # 4. Scale features using the saved scaler
        full_features_scaled = scaler.transform(full_features)
        
        # 5. Predict UPDRS
        predicted_updrs = model.predict(full_features_scaled)[0]
        
        return float(predicted_updrs)
    
    except Exception as e:
        st.error(f"Prediction Error: Could not process audio or predict. Details: {e}")
        return None
    
    finally:
        # 6. Cleanup
        if final_wav_path and os.path.exists(final_wav_path):
            os.remove(final_wav_path)


# --- 3. Streamlit UI and Execution ---

# Sidebar for User Demographics
st.sidebar.header("📋 User Profile")
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=60, step=1)
sex_label = st.sidebar.selectbox("Sex", options=["Male", "Female"])
sex = 0 if sex_label == "Male" else 1  # 0: Male, 1: Female matching dataset

st.title("🎙️ Voice Analysis for Parkinson's Severity")
st.markdown("Record a short 'ah' sound. The audio will be analyzed alongside your profile to estimate Parkinson's symptom severity (UPDRS).")


# --- COMPONENT CALL (FINAL & BARE FIX) ---
# Call st_audiorec with NO arguments to avoid the TypeError
audio_bytes = st_audiorec()
# ------------------------------------------


# Only run prediction when the component has returned data (which is raw WAV bytes)
if audio_bytes is not None:
    st.info("✅ Audio recording complete. Sending to model...")

    with st.spinner('Running AI Model...'):
        # Pass raw bytes directly to the prediction function
        predicted_updrs = predict_on_audio(audio_bytes, age, sex)
    
    # --- Display Result ---
    if predicted_updrs is not None:
        st.subheader("📊 Prediction Results")
        
        # Show predicted UPDRS score
        st.metric("Estimated Total UPDRS Score", f"{predicted_updrs:.2f}")
        
        # UPDRS usually ranges from 0 to 176. Define approximate visual levels.
        if predicted_updrs < 30:
            severity = "Mild Symptoms / Early Stage"
            st.success(f"**Severity Level: {severity}**")
        elif predicted_updrs < 60:
            severity = "Moderate Symptoms / Moderate Stage"
            st.warning(f"**Severity Level: {severity}**")
        else:
            severity = "Severe Symptoms / Advanced Stage"
            st.error(f"**Severity Level: {severity}**")
            
        # Display progress bar showing score relative to maximum (176)
        normalized_updrs = min(max(predicted_updrs / 176.0, 0.0), 1.0)
        st.progress(normalized_updrs)
        
        st.info("ℹ️ **What is UPDRS?** The Unified Parkinson's Disease Rating Scale (UPDRS) is a tool used to monitor the course of Parkinson's disease. A higher score indicates a higher level of symptoms and disability (maximum score is 176).")
        st.warning("⚠️ **Disclaimer:** This is a screening tool, not a medical diagnosis. Consult a professional.")
        
        # Optional: Playback the recorded audio
        st.markdown("---")
        st.caption("Review your recording:")
        st.audio(audio_bytes, format='audio/wav')