# 🎙️ Voice-Based Parkinson’s Disease Detection

## 📌 Project Overview
This project detects the risk of Parkinson’s disease using voice recordings. A short speech sample is analyzed to extract acoustic features, which are then evaluated using a trained machine learning model to predict Parkinson’s risk.

⚠️ *This is a screening and educational tool, not a medical diagnosis.*

---

## 🧠 How It Works
1. User records a short sustained vowel sound (“ah”) via the web app  
2. Audio features are extracted from the `.wav` file  
3. Features are scaled and passed to a trained Random Forest model  
4. The model outputs a prediction and confidence score  

---

---

## 📊 Dataset
- **Source:** UCI Parkinson’s Disease Dataset  
- **Features:** Jitter, shimmer, pitch, harmonic-to-noise ratio  
- **Target Label:**
  - `0` → Healthy  
  - `1` → Parkinson’s Disease  

---

## ⚙️ Machine Learning Pipeline
- Feature Scaling: `StandardScaler`
- Data Balancing: `SMOTE`
- Model: `Random Forest Classifier`
- Output: Risk label + confidence score

---

## 🎧 Audio Feature Extraction
Audio features are extracted from voice recordings using:
- **Librosa**
- **Praat-Parselmouth**

These features align with those used in clinical Parkinson’s voice analysis research.

## 🚀 Running the Application

### 1️⃣ Install Dependencies

pip install -r requirements.txt
### 2️⃣ Run the Streamlit App

streamlit run app.py
### 3️⃣ Use the App

Record a short “ah” sound

View prediction and confidence

Listen to the recorded audio 

Thank You :)
