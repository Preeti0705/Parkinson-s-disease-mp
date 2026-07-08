import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
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

# Load DL
checkpoint = torch.load('models/ensemble_models.pt', map_location=DEVICE, weights_only=False)
cnn = CNN1D(checkpoint['input_dim']).to(DEVICE)
cnn.load_state_dict(checkpoint['cnn_state_dict'])
cnn.eval()

lstm = LSTMModel(checkpoint['input_dim']).to(DEVICE)
lstm.load_state_dict(checkpoint['lstm_state_dict'])
lstm.eval()

# Load ML
ml = joblib.load('models/ml_ensemble.pkl')

# Evaluate
X_t = torch.FloatTensor(X_red).to(DEVICE)
with torch.no_grad():
    p_cnn = cnn(X_t.unsqueeze(1)).cpu().numpy()
    p_lstm = lstm(X_t.unsqueeze(2)).cpu().numpy()

p_rf = ml['rf'].predict_proba(X_red)[:, 1]
p_hgb = ml['hgb'].predict_proba(X_red)[:, 1]
p_svm = ml['svm'].predict_proba(X_red)[:, 1]

models = {
    'CNN1D': p_cnn,
    'BiLSTM': p_lstm,
    'RandomForest': p_rf,
    'HistGBM': p_hgb,
    'SVM': p_svm
}

print('Metrics of the saved best fold models on the full dataset (includes training data):')
for name, p in models.items():
    auc = roc_auc_score(y, p)
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='weighted')
    print(f'{name:<15} -> Accuracy: {acc*100:.2f}%, F1: {f1:.4f}, AUC: {auc:.4f}')

# Output weights
weights = checkpoint.get('weights')
if weights is not None:
    print('\nEnsemble weights based on validation AUC in the best fold:')
    for name, w in zip(['CNN1D', 'BiLSTM', 'RandomForest', 'HistGBM', 'SVM'], weights):
        print(f'{name:<15} -> Weight: {w:.4f}')
