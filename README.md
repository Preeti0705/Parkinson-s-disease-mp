# 🧠 Parkinson’s Disease Voice Detection — ML Research Pipeline

## 📌 Project Overview
This project demonstrates an advanced Machine Learning pipeline that detects Parkinson’s disease from 753 acoustic voice features. To ensure methodological correctness and avoid training-inference mismatch, the application operates as a **Research Demonstration Tool** where users upload CSV patient data containing clinical acoustic features for live inference.

⚠️ *This is a screening and educational tool, not a medical diagnosis.*

---

## 🚀 The Hybrid Ensemble Pipeline

This project employs a robust **5-model AUC-weighted Soft Voting Ensemble**. 

### Why is this Ensemble Technique the Best Approach?
Detecting Parkinson's from voice requires analyzing 753 highly complex, multi-dimensional acoustic features (Jitter, Shimmer, MFCCs, Wavelets, etc.). No single model is perfectly suited for all of these:
1. **Residual CNN1D:** Excellent at automatically extracting local, spatial feature clusters across the 1D acoustic sequence.
2. **BiLSTM with Self-Attention:** Designed to capture sequential dependencies and temporally relevant "windows" of acoustic biomarkers.
3. **Random Forest & HistGradientBoosting:** Tree-based models are extremely robust against non-linear tabular data and less prone to overfitting on small datasets.
4. **Calibrated SVM:** Finds optimal hyperplanes in the high-dimensional PCA space.

By combining them via **AUC-weighted soft voting**, the ensemble naturally mitigates the weaknesses of any individual model. Models that performed better during cross-validation (higher AUC) are given more voting power. Finally, we use **Youden's J statistic** to find the absolute optimal decision threshold (rather than defaulting to 0.5), which is crucial for imbalanced medical datasets.

---

## ⚙️ Data Preprocessing & Flow
1. **Data Load:** 753 acoustic features per patient (`data/pd_speech_features.csv`).
2. **StandardScaler:** Zero-mean, unit-variance scaling.
3. **Boruta Feature Selection:** Statistically reduces noise (753 → 131 features).
4. **PCA:** Reduces dimensionality while retaining 99% variance (131 → ~69 components).
5. **ADASYN:** Synthetic oversampling (applied *only* to training folds).
6. **Inference:** The exact same scaler, Boruta mask, and PCA components are applied to user-uploaded CSVs during deployment to ensure 100% parity.

---

## 💻 Running the Application

### 1️⃣ Install Dependencies
Ensure you are using the virtual environment:
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Train the Models (Optional)
Run the full 5-fold Stratified Cross-Validation pipeline. This will train all models, evaluate them independently, and save the deployment artifacts.
```powershell
python src/advanced_model.py
```

### 3️⃣ Evaluate Saved Models
Run a rigorous independent evaluation on the final deployment artifacts:
```powershell
python evaluate_saved_models.py
```

### 4️⃣ Run the Web Application
Launch the Streamlit interface:
```powershell
streamlit run app.py
```

- Upload a CSV containing patient data (must have the same 753 feature columns).
- View instant predictions, risk scores, and the individual probability breakdown for all 5 models.
- Download the results as a CSV.
