import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import sys
import os

from src.advanced_model import CNN1D, LSTMModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv('data/pd_speech_features.csv', header=1)
X = df.drop(columns=['id', 'class']).values
y = df['class'].values

# Load prep
scaler = joblib.load('models/scaler.pkl')
boruta_mask = joblib.load('models/boruta_mask.pkl')
pca = joblib.load('models/pca.pkl')

X_scaled = scaler.transform(X)
X_sel = X_scaled[:, boruta_mask]
X_red = pca.transform(X_sel)

# Load DL models
checkpoint = torch.load('models/ensemble_models.pt', map_location=DEVICE, weights_only=False)
cnn = CNN1D(checkpoint['input_dim']).to(DEVICE)
cnn.load_state_dict(checkpoint['cnn_state_dict'])
cnn.eval()

lstm = LSTMModel(checkpoint['input_dim']).to(DEVICE)
lstm.load_state_dict(checkpoint['lstm_state_dict'])
lstm.eval()

# Load ML models
ml = joblib.load('models/ml_ensemble.pkl')

# Evaluate probabilities
X_t = torch.FloatTensor(X_red).to(DEVICE)
with torch.no_grad():
    p_cnn = cnn(X_t.unsqueeze(1)).cpu().numpy()
    p_lstm = lstm(X_t.unsqueeze(2)).cpu().numpy()

p_rf = ml['rf'].predict_proba(X_red)[:, 1]
p_hgb = ml['hgb'].predict_proba(X_red)[:, 1]
p_svm = ml['svm'].predict_proba(X_red)[:, 1]

# 1. Individual metrics for requested models
models_to_compare = {
    'CNN1D': p_cnn,
    'BiLSTM': p_lstm,
    'RandomForest': p_rf
}

print("="*65)
print("1. INDIVIDUAL MODEL METRICS (CNN1D, BiLSTM, RandomForest)")
print("="*65)
# print("Understanding the Confusion Matrix:")
# print(" TP = True Positives (Caught Parkinson's)")
# print(" FN = False Negatives (Missed Parkinson's - CRITICAL ERROR)")
# print(" TN = True Negatives (Correctly identified Healthy)")
# print(" FP = False Positives (False Alarm - predicted PD but healthy)")
print("-" * 65)

for name, p in models_to_compare.items():
    auc = roc_auc_score(y, p)
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='weighted')
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    
    print(f"--- {name} ---")
    print(f" Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f} | F1-Score: {f1:.4f}")
    print(f" Caught PD (TP): {tp:>3}  |  Missed PD (FN): {fn:>3} !!")
    print(f" Healthy (TN):   {tn:>3}  |  False Alarm (FP): {fp:>3}")
    print()

print("="*65)
print("2. WHY ENSEMBLING IS BETTER: CORRELATION & DIVERSITY")
print("="*65)
df_preds = pd.DataFrame({
    'CNN1D': p_cnn,
    'BiLSTM': p_lstm,
    'RandomForest': p_rf
})
print("Prediction Correlation Matrix:")
print(df_preds.corr().round(3))
# print("\nINSIGHT:")
# print("The correlation between these models is far below 1.0 (e.g. CNN1D and BiLSTM")
# print("have a correlation around ~0.7-0.8). This mathematical variance proves that")
# print("they learn completely different patterns. When CNN1D makes a mistake,")
# print("RandomForest or BiLSTM likely caught it, allowing them to outvote the error.")
# print()

print("="*65)
print("3. ENSEMBLE METRICS (The Result of Teamwork)")
print("="*65)

# A simple ensemble of the 3 requested models
p_ensemble_3 = (p_cnn + p_lstm + p_rf) / 3.0
pred_ens3 = (p_ensemble_3 >= 0.5).astype(int)
acc_e3 = accuracy_score(y, pred_ens3)
auc_e3 = roc_auc_score(y, p_ensemble_3)
f1_e3 = f1_score(y, pred_ens3, average='weighted')
tn3, fp3, fn3, tp3 = confusion_matrix(y, pred_ens3).ravel()

print("--- Simple 3-Model Ensemble (CNN1D + BiLSTM + RandomForest) ---")
print(f" Accuracy: {acc_e3*100:.2f}% | ROC-AUC: {auc_e3:.4f} | F1-Score: {f1_e3:.4f}")
print(f" Caught PD (TP): {tp3:>3}  |  Missed PD (FN): {fn3:>3}")
print(f" Healthy (TN):   {tn3:>3}  |  False Alarm (FP): {fp3:>3}")
# print("\nINSIGHT:")
# print("Notice how the ensemble balances the weaknesses of individual models.")
# print("It provides a more stable, robust set of predictions by averaging out")
# print("individual model overconfidence.")
# print()

# Full 5-model weighted ensemble with optimal threshold
weights = checkpoint.get('weights', [0.2]*5)
p_full_ensemble = (p_cnn*weights[0] + p_lstm*weights[1] + p_rf*weights[2] + p_hgb*weights[3] + p_svm*weights[4])
optimal_threshold = checkpoint.get('threshold', 0.5)

pred_full = (p_full_ensemble >= optimal_threshold).astype(int)
acc_f = accuracy_score(y, pred_full)
auc_f = roc_auc_score(y, p_full_ensemble)
tn_f, fp_f, fn_f, tp_f = confusion_matrix(y, pred_full).ravel()

# print("--- Full 5-Model Weighted Ensemble with Optimal Threshold ---")
# print(f" Accuracy: {acc_f*100:.2f}% | ROC-AUC: {auc_f:.4f}")
# print(f" Caught PD (TP): {tp_f:>3}  |  Missed PD (FN): {fn_f:>3} !!")
# print(f" Healthy (TN):   {tn_f:>3}  |  False Alarm (FP): {fp_f:>3}")
# print("\nFINAL MEDICAL JUSTIFICATION:")
# print("By using all 5 models and shifting from the default 0.5 threshold to")
# print(f"the mathematically optimal threshold of {optimal_threshold:.3f}, the ensemble")
# print("drastically reduces 'False Negatives' (Missed PD). In healthcare, missing")
# print("a sick patient is far worse than a false alarm. Ensembling smooths the")
# print("probability curve so we can pinpoint this exact threshold.")
