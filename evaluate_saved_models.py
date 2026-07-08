import os
import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.advanced_model import CNN1D, LSTMModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_models():
    print("=" * 70)
    print("  DEPLOYMENT EVALUATION OF FINAL SAVED MODELS")
    print("=" * 70)
    
    # 1. Load Data
    csv_path = 'data/pd_speech_features.csv'
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, header=1)
    X = df.drop(columns=['id', 'class']).values
    y = df['class'].values
    
    # 2. Load Preprocessing
    print("Loading preprocessing pipeline...")
    scaler = joblib.load('models/scaler.pkl')
    boruta_mask = joblib.load('models/boruta_mask.pkl')
    pca = joblib.load('models/pca.pkl')
    
    print("Applying preprocessing...")
    X_scaled = scaler.transform(X)
    X_sel = X_scaled[:, boruta_mask]
    X_red = pca.transform(X_sel)
    
    # 3. Load Models
    print("Loading trained models...")
    checkpoint = torch.load('models/ensemble_models.pt', map_location=DEVICE, weights_only=False)
    input_dim = checkpoint['input_dim']
    optimal_threshold = checkpoint.get('threshold', 0.5)
    
    cnn = CNN1D(input_dim).to(DEVICE)
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    cnn.eval()
    
    lstm = LSTMModel(input_dim).to(DEVICE)
    lstm.load_state_dict(checkpoint['lstm_state_dict'])
    lstm.eval()
    
    ml_models = joblib.load('models/ml_ensemble.pkl')
    
    # 4. Generate Probabilities
    print(f"\nEvaluating with Optimal Threshold: {optimal_threshold:.4f}\n")
    X_t = torch.FloatTensor(X_red).to(DEVICE)
    with torch.no_grad():
        p_cnn = cnn(X_t.unsqueeze(1)).cpu().numpy()
        p_lstm = lstm(X_t.unsqueeze(2)).cpu().numpy()
        
    p_rf = ml_models['rf'].predict_proba(X_red)[:, 1]
    p_hgb = ml_models['hgb'].predict_proba(X_red)[:, 1]
    p_svm = ml_models['svm'].predict_proba(X_red)[:, 1]
    
    weights = checkpoint.get('weights', [0.2]*5)
    w = np.array(weights)
    w = w / w.sum()
    probas = np.stack([p_cnn, p_lstm, p_rf, p_hgb, p_svm], axis=1)
    p_ens = (probas * w).sum(axis=1)
    
    probas_dict = {
        'CNN1D': p_cnn,
        'BiLSTM': p_lstm,
        'Random Forest': p_rf,
        'HistGradientBoosting': p_hgb,
        'SVM': p_svm,
        'Ensemble': p_ens
    }
    
    # 5. Calculate Metrics
    print(f"| {'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|")
    
    for name, p in probas_dict.items():
        try: auc = roc_auc_score(y, p)
        except: auc = 0.5
        
        pred = (p >= optimal_threshold).astype(int)
        acc = accuracy_score(y, pred)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, average='weighted')
        
        print(f"| {name:<20} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}     | {auc:.4f}     |")

if __name__ == '__main__':
    evaluate_models()
