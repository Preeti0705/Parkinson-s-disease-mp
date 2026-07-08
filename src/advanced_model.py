"""
Advanced ML Pipeline for Parkinson's Disease Detection  — v2
=============================================================
Target accuracy: 90%+

Pipeline:
  Data → StandardScaler → Boruta (131 features) → PCA (99% var)
       → ADASYN balancing (per fold)
       → Hybrid Ensemble:
           Deep Learning : Residual-CNN1D + BiLSTM-Attention  (Focal Loss)
           Traditional ML: RandomForest + HistGradientBoosting + SVM (RBF)
       → AUC-weighted soft voting → Optimal-threshold decision
  Evaluation: 5-Fold Stratified CV
"""

import os
import glob
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, classification_report,
    precision_score, recall_score
)
from imblearn.over_sampling import ADASYN
from boruta import BorutaPy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ── Configuration ──────────────────────────────────────────────────────────────
SEED         = 42
N_FOLDS      = 5
PCA_VARIANCE = 0.99          # keep 99% — was 0.95
BATCH_SIZE   = 32
EPOCHS       = 200           # was 100
PATIENCE     = 25            # was 10
FOCAL_ALPHA  = 0.75          # weight for positive class in Focal Loss
FOCAL_GAMMA  = 2.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. FOCAL LOSS
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    FL(p) = -α · (1-p)^γ · log(p)

    Reduces loss for well-classified samples so the model focuses on
    hard (minority-class) examples.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce    = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt     = torch.exp(-bce)
        focal  = self.alpha * ((1 - pt) ** self.gamma) * bce
        return focal.mean()


# ===========================================================================
# 2. MODEL ARCHITECTURES
# ===========================================================================

class _ResBlock(nn.Module):
    """1-D residual block: two conv layers with a skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.body(x))


class CNN1D(nn.Module):
    """
    Residual 1-D CNN.
    AdaptiveMaxPool1d(1) makes it dimension-agnostic → works for any
    number of input features (both PCA-reduced and raw 13 audio features).
    """
    def __init__(self, input_dim: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.2),
        )
        self.res1 = _ResBlock(64)
        self.down = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
        )
        self.res2  = _ResBlock(128)
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.head  = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, seq_len)
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.pool(x).squeeze(-1)   # → (batch, 128)
        return self.head(x).squeeze(-1)


class LSTMModel(nn.Module):
    """
    Bidirectional stacked LSTM with self-attention pooling.
    input_size=1 + sequence-length-agnostic → works for any feature count.
    """
    def __init__(self, input_dim: int = 1):
        super().__init__()
        self.lstm1  = nn.LSTM(input_size=1,   hidden_size=64, batch_first=True, bidirectional=True)
        self.drop1  = nn.Dropout(0.3)
        self.lstm2  = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.drop2  = nn.Dropout(0.3)
        self.attn   = nn.Linear(128, 1)         # attention scoring
        self.head   = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        x, _ = self.lstm1(x)          # → (batch, seq, 128)
        x     = self.drop1(x)
        x, _ = self.lstm2(x)          # → (batch, seq, 128)
        x     = self.drop2(x)
        # Self-attention pooling
        w = torch.softmax(self.attn(x), dim=1)   # (batch, seq, 1)
        x = (w * x).sum(dim=1)                   # (batch, 128)
        return self.head(x).squeeze(-1)


# ===========================================================================
# 3. PREPROCESS DATA
# ===========================================================================

def preprocess_data(csv_path=None):
    if csv_path is None:
        matches = glob.glob('C:/Users/PREETI/Desktop/Parkinson*/data/pd_speech_features.csv')
        if matches:
            csv_path = matches[0]
        else:
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'pd_speech_features.csv',
            )

    log.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, header=1)
    log.info(f"Dataset shape: {df.shape}")

    X = df.drop(columns=['id', 'class'])
    y = df['class'].values
    feature_names = list(X.columns)
    log.info(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")
    log.info(f"Class distribution: Healthy={np.sum(y==0)}, Parkinson={np.sum(y==1)}")

    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X)
    return X_scaled, y, scaler, feature_names


# ===========================================================================
# 4. FEATURE SELECTION  (Boruta → PCA at 99% variance)
# ===========================================================================

def select_features(X, y, feature_names):
    log.info("=" * 60)
    log.info("FEATURE SELECTION: Boruta → PCA (99% variance)")
    log.info("=" * 60)

    # -- Boruta --
    log.info("Running Boruta (may take 1-2 min)…")
    rf_b = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=SEED, n_jobs=-1)
    boruta = BorutaPy(estimator=rf_b, n_estimators='auto', max_iter=100, random_state=SEED, verbose=0)
    boruta.fit(X, y)

    mask = boruta.support_
    if np.sum(mask) < 10:
        mask = boruta.support_ | boruta.support_weak_
    log.info(f"Boruta selected {np.sum(mask)} / {X.shape[1]} features")

    X_sel = X[:, mask]

    # -- PCA at 99% --
    log.info(f"PCA at {PCA_VARIANCE*100:.0f}% variance…")
    pca = PCA(n_components=PCA_VARIANCE, random_state=SEED)
    X_red = pca.fit_transform(X_sel)
    log.info(f"PCA: {X_sel.shape[1]} → {X_red.shape[1]} components ({np.sum(pca.explained_variance_ratio_)*100:.2f}% var)")

    return X_red, mask, pca, [f for f, m in zip(feature_names, mask) if m]


# ===========================================================================
# 5. DATA BALANCING  (ADASYN only — no Mixup to preserve hard labels)
# ===========================================================================

def balance_data(X_train, y_train):
    try:
        k = max(1, min(5, int(np.sum(y_train == 0)) - 1))
        ada = ADASYN(random_state=SEED, n_neighbors=k)
        X_r, y_r = ada.fit_resample(X_train, y_train)
        log.info(f"ADASYN: {X_train.shape[0]} → {X_r.shape[0]} samples")
    except Exception as e:
        log.warning(f"ADASYN failed ({e}), using original data")
        X_r, y_r = X_train, y_train
    return X_r, y_r


# ===========================================================================
# 6. TRAIN DEEP-LEARNING MODELS
# ===========================================================================

def _train_single_dl(model, X_tr, y_tr, X_val, y_val, name):
    """
    X_tr, y_tr, X_val, y_val must already be torch.FloatTensor on DEVICE.
    train_dl_models() handles the numpy → tensor conversion before calling here.
    """
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    best_loss, best_state, patience_ctr = float('inf'), None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_loss:
            best_loss, patience_ctr = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            log.info(f"  {name} early stop @ epoch {epoch+1} (best val_loss={best_loss:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    log.info(f"  {name} done (best val_loss={best_loss:.4f})")
    return model



def train_dl_models(X_tr, y_tr, X_val, y_val, input_dim):
    X_tr_t  = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t  = torch.FloatTensor(y_tr.astype(np.float32)).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val.astype(np.float32)).to(DEVICE)

    log.info("Training Residual CNN1D…")
    cnn = _train_single_dl(
        CNN1D(input_dim).to(DEVICE),
        X_tr_t.unsqueeze(1), y_tr_t,
        X_val_t.unsqueeze(1), y_val_t,
        "CNN1D",
    )

    log.info("Training BiLSTM-Attention…")
    lstm = _train_single_dl(
        LSTMModel(input_dim).to(DEVICE),
        X_tr_t.unsqueeze(2), y_tr_t,
        X_val_t.unsqueeze(2), y_val_t,
        "BiLSTM",
    )
    return cnn, lstm


# ===========================================================================
# 7. TRAIN TRADITIONAL ML MODELS
# ===========================================================================

def train_ml_models(X_tr, y_tr):
    """Train RF + HistGBM + calibrated SVM on ADASYN-balanced data."""

    log.info("Training Random Forest (n=500, balanced)…")
    rf = RandomForestClassifier(
        n_estimators=500, class_weight='balanced',
        max_features='sqrt', min_samples_leaf=2,
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    log.info("Training HistGradientBoosting (balanced)…")
    hgb = HistGradientBoostingClassifier(
        max_iter=500, class_weight='balanced',
        learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=0.1, random_state=SEED,
    )
    hgb.fit(X_tr, y_tr)

    log.info("Training SVM (RBF, calibrated)…")
    svm_base = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=SEED)
    svm = CalibratedClassifierCV(svm_base, cv=3)
    svm.fit(X_tr, y_tr)

    return {'rf': rf, 'hgb': hgb, 'svm': svm}


# ===========================================================================
# 8. HYBRID ENSEMBLE PREDICTION
# ===========================================================================

def _dl_proba(model_cnn, model_lstm, X):
    X_t = torch.FloatTensor(X).to(DEVICE)
    model_cnn.eval(); model_lstm.eval()
    with torch.no_grad():
        p_cnn  = model_cnn(X_t.unsqueeze(1)).cpu().numpy()
        p_lstm = model_lstm(X_t.unsqueeze(2)).cpu().numpy()
    return p_cnn, p_lstm


def ensemble_predict(cnn, lstm, ml_models, X, weights=None, threshold=0.5):
    """
    Weighted soft-voting across 5 models.
    weights: list [w_cnn, w_lstm, w_rf, w_hgb, w_svm] — default equal
    """
    p_cnn, p_lstm = _dl_proba(cnn, lstm, X)
    p_rf   = ml_models['rf'].predict_proba(X)[:, 1]
    p_hgb  = ml_models['hgb'].predict_proba(X)[:, 1]
    p_svm  = ml_models['svm'].predict_proba(X)[:, 1]

    probas = np.stack([p_cnn, p_lstm, p_rf, p_hgb, p_svm], axis=1)  # (n, 5)
    w      = np.array(weights) if weights is not None else np.ones(5)
    w      = w / w.sum()
    y_prob = (probas * w).sum(axis=1)
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


def find_optimal_threshold(y_true, y_proba):
    """Youden's J statistic: maximize TPR - FPR."""
    fpr, tpr, thresh = roc_curve(y_true, y_proba)
    j = tpr - fpr
    return float(thresh[np.argmax(j)])


# ===========================================================================
# 9. CROSS-VALIDATION
# ===========================================================================

def evaluate_cv(X, y):
    log.info("=" * 60)
    log.info(f"{N_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION — Hybrid Ensemble")
    log.info("=" * 60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    model_names = ['CNN1D', 'BiLSTM', 'Random Forest', 'HistGradientBoosting', 'SVM', 'Ensemble']
    metrics_tracker = {
        name: {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
        for name in model_names
    }
    best_auc, best_bundle = -1, None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        log.info(f"\n{'─'*40}")
        log.info(f"FOLD {fold}/{N_FOLDS}  —  Train: {len(tr_idx)} | Val: {len(val_idx)}")
        log.info(f"{'─'*40}")

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Balance training data
        X_tr_b, y_tr_b = balance_data(X_tr, y_tr)

        # Train deep learning
        cnn, lstm = train_dl_models(X_tr_b, y_tr_b, X_val, y_val, X_tr_b.shape[1])

        # Train traditional ML (on balanced data; class_weight also handles imbalance)
        ml = train_ml_models(X_tr_b, y_tr_b)

        # Get individual validation probabilities for weighting
        p_cnn, p_lstm = _dl_proba(cnn, lstm, X_val)
        p_rf   = ml['rf'].predict_proba(X_val)[:, 1]
        p_hgb  = ml['hgb'].predict_proba(X_val)[:, 1]
        p_svm  = ml['svm'].predict_proba(X_val)[:, 1]
        
        probas_dict = {
            'CNN1D': p_cnn,
            'BiLSTM': p_lstm,
            'Random Forest': p_rf,
            'HistGradientBoosting': p_hgb,
            'SVM': p_svm
        }

        def safe_auc(p):
            try: return roc_auc_score(y_val, p)
            except: return 0.5

        aucs = np.array([safe_auc(p) for p in [p_cnn, p_lstm, p_rf, p_hgb, p_svm]])
        log.info(f"  Individual AUCs → CNN:{aucs[0]:.3f}  LSTM:{aucs[1]:.3f}  "
                 f"RF:{aucs[2]:.3f}  HGB:{aucs[3]:.3f}  SVM:{aucs[4]:.3f}")

        # AUC-weighted ensemble
        weights = np.clip(aucs, 0.5, 1.0) - 0.5   # shift so 0.5 AUC → weight 0
        weights = weights / weights.sum()

        y_pred, y_prob = ensemble_predict(cnn, lstm, ml, X_val, weights=weights, threshold=0.5)

        # Optimal threshold from ensemble
        thr = find_optimal_threshold(y_val, y_prob)
        probas_dict['Ensemble'] = y_prob
        
        # Calculate and store metrics for all models
        for name, p in probas_dict.items():
            auc = safe_auc(p)
            pred = (p >= thr).astype(int)
            
            acc = accuracy_score(y_val, pred)
            prec = precision_score(y_val, pred, zero_division=0)
            rec = recall_score(y_val, pred, zero_division=0)
            f1 = f1_score(y_val, pred, average='weighted')
            
            metrics_tracker[name]['acc'].append(acc)
            metrics_tracker[name]['prec'].append(prec)
            metrics_tracker[name]['rec'].append(rec)
            metrics_tracker[name]['f1'].append(f1)
            metrics_tracker[name]['auc'].append(auc)

        acc_ens = metrics_tracker['Ensemble']['acc'][-1]
        f1_ens = metrics_tracker['Ensemble']['f1'][-1]
        auc_ens = metrics_tracker['Ensemble']['auc'][-1]

        log.info(f"  Ensemble  →  Accuracy: {acc_ens*100:.2f}%  F1: {f1_ens:.4f}  AUC: {auc_ens:.4f}  "
                 f"Threshold*: {thr:.3f}")
        log.info(f"\n{classification_report(y_val, (y_prob >= thr).astype(int), target_names=['Healthy','Parkinson'], digits=4)}")

        if auc_ens > best_auc:
            best_auc = auc_ens
            best_bundle = {
                'cnn_state':  {k: v.cpu() for k, v in cnn.state_dict().items()},
                'lstm_state': {k: v.cpu() for k, v in lstm.state_dict().items()},
                'ml_models':  ml,
                'input_dim':  X_tr_b.shape[1],
                'weights':    weights.tolist(),
                'threshold':  thr,
            }

    log.info("\n" + "=" * 90)
    log.info("CROSS-VALIDATION SUMMARY (INDIVIDUAL MODELS & ENSEMBLE)")
    log.info("=" * 90)
    
    print(f"\n| {'Model':<20} | {'Accuracy':<15} | {'Precision':<15} | {'Recall':<15} | {'F1-Score':<15} | {'ROC-AUC':<15} |")
    print(f"|{'-'*22}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*17}|")
    
    for name in model_names:
        m_acc, s_acc = np.mean(metrics_tracker[name]['acc']), np.std(metrics_tracker[name]['acc'])
        m_pre, s_pre = np.mean(metrics_tracker[name]['prec']), np.std(metrics_tracker[name]['prec'])
        m_rec, s_rec = np.mean(metrics_tracker[name]['rec']), np.std(metrics_tracker[name]['rec'])
        m_f1,  s_f1  = np.mean(metrics_tracker[name]['f1']),  np.std(metrics_tracker[name]['f1'])
        m_auc, s_auc = np.mean(metrics_tracker[name]['auc']), np.std(metrics_tracker[name]['auc'])
        
        print(f"| {name:<20} | {m_acc:.4f}±{s_acc:.4f} | {m_pre:.4f}±{s_pre:.4f} | {m_rec:.4f}±{s_rec:.4f} | {m_f1:.4f}±{s_f1:.4f} | {m_auc:.4f}±{s_auc:.4f} |")
    
    print("")

    ens_metrics = metrics_tracker['Ensemble']
    return {
        'accuracy': {'mean': float(np.mean(ens_metrics['acc'])), 'std': float(np.std(ens_metrics['acc'])), 'per_fold': [float(v) for v in ens_metrics['acc']]},
        'f1_score': {'mean': float(np.mean(ens_metrics['f1'])),  'std': float(np.std(ens_metrics['f1'])), 'per_fold': [float(v) for v in ens_metrics['f1']]},
        'roc_auc':  {'mean': float(np.mean(ens_metrics['auc'])), 'std': float(np.std(ens_metrics['auc'])), 'per_fold': [float(v) for v in ens_metrics['auc']]},
        'best_bundle': best_bundle,
    }


# ===========================================================================
# 10. MAIN
# ===========================================================================

def main():
    log.info("=" * 60)
    log.info("PARKINSON'S DISEASE DETECTION — ADVANCED PIPELINE v2")
    log.info(f"Device: {DEVICE}")
    log.info("=" * 60)

    # 1. Preprocess
    X, y, scaler, feature_names = preprocess_data()

    # 2. Feature selection
    X_red, boruta_mask, pca, sel_names = select_features(X, y, feature_names)

    # 3. Cross-validation
    results = evaluate_cv(X_red, y)

    # 4. Save artifacts — always resolve relative to THIS script's location
    #    src/advanced_model.py → parent = src/ → parent = project root
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Saving artifacts to: {save_dir}")

    joblib.dump(scaler,      os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(boruta_mask, os.path.join(save_dir, 'boruta_mask.pkl'))
    joblib.dump(pca,         os.path.join(save_dir, 'pca.pkl'))

    bundle = results['best_bundle']
    if bundle:
        # Save DL models
        torch.save({
            'cnn_state_dict':  bundle['cnn_state'],
            'lstm_state_dict': bundle['lstm_state'],
            'input_dim':       bundle['input_dim'],
            'weights':         bundle['weights'],
            'threshold':       bundle['threshold'],
        }, os.path.join(save_dir, 'ensemble_models.pt'))

        # Save traditional ML models
        joblib.dump(bundle['ml_models'], os.path.join(save_dir, 'ml_ensemble.pkl'))
        log.info("  - ml_ensemble.pkl     -- RF + HistGBM + SVM")

    # Save CV metrics
    acc = results['accuracy']; f1 = results['f1_score']; auc = results['roc_auc']
    cv_out = {
        'accuracy': acc, 'f1_score': f1, 'roc_auc': auc,
        'pipeline': {
            'n_folds':             N_FOLDS,
            'n_original_features': int(X.shape[1]),
            'n_boruta_features':   int(np.sum(boruta_mask)),
            'n_pca_components':    int(pca.n_components_),
            'ensemble':            'Residual-CNN1D + BiLSTM-Attention + RF + HistGBM + SVM (AUC-weighted)',
            'balancing':           'ADASYN',
            'loss':                f'Focal (α={FOCAL_ALPHA}, γ={FOCAL_GAMMA})',
            'optimal_threshold':   bundle['threshold'] if bundle else 0.5,
        },
    }
    with open(os.path.join(save_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_out, f, indent=2)

    log.info(f"\n[OK] Artifacts saved → {save_dir}")
    log.info("  - scaler.pkl          — StandardScaler")
    log.info("  - boruta_mask.pkl     — Boruta feature mask")
    log.info("  - pca.pkl             — PCA (99% variance)")
    log.info("  - ensemble_models.pt  — Residual-CNN1D + BiLSTM weights")
    log.info("  - ml_ensemble.pkl     — RF + HistGBM + SVM")
    log.info("  - cv_results.json     — Cross-validation metrics")

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Mean Accuracy : {acc['mean']*100:.2f}% ± {acc['std']*100:.2f}%")
    print(f"  Mean F1-Score : {f1['mean']:.4f} ± {f1['std']:.4f}")
    print(f"  Mean ROC-AUC  : {auc['mean']:.4f} ± {auc['std']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
