"""
Advanced ML Pipeline for Parkinson's Disease Detection
=======================================================
Pipeline: ADASYN + Mixup -> Boruta + PCA -> CNN1D-LSTM Ensemble -> 5-Fold Stratified CV

Dataset: pd_speech_features.csv (756 samples x 753 features, binary classification)
"""

import os
import glob
import logging
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import ADASYN
from boruta import BorutaPy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -- Configuration --------------------------------------------------------
SEED = 42
N_FOLDS = 5
PCA_VARIANCE = 0.95
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10  # Early stopping patience
MIXUP_ALPHA = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Reproducibility -------------------------------------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Suppress noisy warnings
warnings.filterwarnings('ignore')

# -- Logging Setup ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. PREPROCESS DATA
# ===========================================================================

def preprocess_data(csv_path=None):
    """Load CSV, drop id column, separate X/y, scale features.

    Returns:
        X_scaled (np.ndarray): Scaled feature matrix
        y (np.ndarray): Binary target array
        scaler (StandardScaler): Fitted scaler
        feature_names (list): Column names of features
    """
    if csv_path is None:
        matches = glob.glob('C:/Users/PREETI/Desktop/Parkinson*/data/pd_speech_features.csv')
        if not matches:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(os.path.dirname(script_dir), 'data', 'pd_speech_features.csv')
        else:
            csv_path = matches[0]

    log.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, header=1)
    log.info(f"Dataset shape: {df.shape}")

    # Drop identifier, separate features and target
    X = df.drop(columns=['id', 'class'])
    y = df['class'].values
    feature_names = list(X.columns)

    log.info(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")
    log.info(f"Class distribution: Healthy={np.sum(y == 0)}, Parkinson={np.sum(y == 1)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_names


# ===========================================================================
# 2. FEATURE SELECTION (Boruta + PCA)
# ===========================================================================

def select_features(X, y, feature_names):
    """Apply Boruta for feature selection, then PCA for dimensionality reduction.

    Returns:
        X_reduced (np.ndarray): Reduced feature matrix
        boruta_mask (np.ndarray): Boolean mask of Boruta-selected features
        pca (PCA): Fitted PCA transformer
        selected_feature_names (list): Names of Boruta-selected features
    """
    log.info("=" * 60)
    log.info("FEATURE SELECTION: Boruta + PCA")
    log.info("=" * 60)

    # -- Boruta Feature Selection --
    log.info("Running Boruta feature selection (this may take a few minutes)...")
    rf_boruta = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=SEED,
        n_jobs=-1
    )
    boruta = BorutaPy(
        estimator=rf_boruta,
        n_estimators='auto',
        max_iter=100,
        random_state=SEED,
        verbose=0
    )
    boruta.fit(X, y)

    boruta_mask = boruta.support_
    n_selected = np.sum(boruta_mask)
    log.info(f"Boruta selected {n_selected} out of {X.shape[1]} features")

    # If Boruta selects too few features, also include tentative ones
    if n_selected < 10:
        boruta_mask = boruta.support_ | boruta.support_weak_
        n_selected = np.sum(boruta_mask)
        log.info(f"Including tentative features -> {n_selected} total")

    selected_feature_names = [f for f, m in zip(feature_names, boruta_mask) if m]
    X_selected = X[:, boruta_mask]

    # -- PCA Dimensionality Reduction --
    log.info(f"Applying PCA with {PCA_VARIANCE * 100:.0f}% variance retention...")
    pca = PCA(n_components=PCA_VARIANCE, random_state=SEED)
    X_reduced = pca.fit_transform(X_selected)

    log.info(f"PCA reduced {X_selected.shape[1]} -> {X_reduced.shape[1]} components")
    log.info(f"Explained variance: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

    return X_reduced, boruta_mask, pca, selected_feature_names


# ===========================================================================
# 3. DATA BALANCING (ADASYN + Mixup)
# ===========================================================================

def balance_data(X_train, y_train):
    """Apply ADASYN oversampling + Mixup augmentation.

    Returns:
        X_balanced (np.ndarray): Balanced + augmented feature matrix
        y_balanced (np.ndarray): Balanced + augmented labels
    """
    log.info("Balancing data with ADASYN...")

    # -- ADASYN --
    try:
        adasyn = ADASYN(random_state=SEED, n_neighbors=min(5, np.sum(y_train == 0) - 1))
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
        log.info(f"ADASYN: {X_train.shape[0]} -> {X_resampled.shape[0]} samples")
    except ValueError as e:
        log.warning(f"ADASYN failed ({e}), using original data")
        X_resampled, y_resampled = X_train, y_train

    # -- Mixup Augmentation --
    log.info("Applying Mixup augmentation...")
    X_mixed, y_mixed = mixup(X_resampled, y_resampled, alpha=MIXUP_ALPHA)

    # Combine original resampled data + mixup data
    X_balanced = np.vstack([X_resampled, X_mixed])
    y_balanced = np.concatenate([y_resampled, y_mixed])

    log.info(f"After Mixup: {X_balanced.shape[0]} total samples")

    return X_balanced, y_balanced


def mixup(X, y, alpha=0.2, n_samples=None):
    """Generate synthetic samples by blending random pairs.

    Args:
        X: Feature matrix
        y: Labels (can be float for soft labels)
        alpha: Beta distribution parameter (smaller = less mixing)
        n_samples: Number of synthetic samples (default: len(X) // 2)

    Returns:
        X_mix, y_mix: Blended synthetic samples
    """
    n = n_samples or len(X) // 2
    indices_a = np.random.randint(0, len(X), n)
    indices_b = np.random.randint(0, len(X), n)
    lam = np.random.beta(alpha, alpha, n).astype(np.float32)

    X_mix = lam[:, None] * X[indices_a] + (1 - lam[:, None]) * X[indices_b]
    y_mix = lam * y[indices_a] + (1 - lam) * y[indices_b]

    return X_mix, y_mix


# ===========================================================================
# 4. MODEL ARCHITECTURES (CNN1D + LSTM)
# ===========================================================================

class CNN1D(nn.Module):
    """1D Convolutional Neural Network for local feature pattern extraction."""

    def __init__(self, input_dim):
        super(CNN1D, self).__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Global Max Pooling
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, 1, features)
        x = self.net(x)          # -> (batch, 128, 1)
        x = x.squeeze(-1)       # -> (batch, 128)
        x = self.classifier(x)  # -> (batch, 1)
        return x.squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM network for capturing sequential dependencies in features."""

    def __init__(self, input_dim):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=1, hidden_size=64,
            batch_first=True, bidirectional=False
        )
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=64, hidden_size=32,
            batch_first=True, bidirectional=False
        )
        self.dropout2 = nn.Dropout(0.3)

        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, features, 1)
        x, _ = self.lstm1(x)     # -> (batch, features, 64)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)     # -> (batch, features, 32)
        x = self.dropout2(x)
        x = x[:, -1, :]         # Take last timestep -> (batch, 32)
        x = self.classifier(x)  # -> (batch, 1)
        return x.squeeze(-1)


# ===========================================================================
# 5. TRAINING
# ===========================================================================

def train_models(X_train, y_train, X_val, y_val):
    """Train CNN1D and LSTM models with early stopping.

    Returns:
        cnn_model: Trained CNN1D
        lstm_model: Trained LSTMModel
    """
    input_dim = X_train.shape[1]

    # Reshape for Conv1D/LSTM: (samples, 1, features) for CNN, (samples, features, 1) for LSTM
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train.astype(np.float32)).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val.astype(np.float32)).to(DEVICE)

    # CNN input: (batch, channels=1, features)
    X_train_cnn = X_train_t.unsqueeze(1)
    X_val_cnn = X_val_t.unsqueeze(1)

    # LSTM input: (batch, seq_len=features, input_size=1)
    X_train_lstm = X_train_t.unsqueeze(2)
    X_val_lstm = X_val_t.unsqueeze(2)

    # Build models
    cnn_model = CNN1D(input_dim).to(DEVICE)
    lstm_model = LSTMModel(input_dim).to(DEVICE)

    # Train each model
    log.info("Training CNN1D...")
    cnn_model = _train_single_model(
        cnn_model, X_train_cnn, y_train_t, X_val_cnn, y_val_t, "CNN1D"
    )

    log.info("Training LSTM...")
    lstm_model = _train_single_model(
        lstm_model, X_train_lstm, y_train_t, X_val_lstm, y_val_t, "LSTM"
    )

    return cnn_model, lstm_model


def _train_single_model(model, X_train, y_train, X_val, y_val, name):
    """Train a single model with early stopping and return best weights."""

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        # -- Train --
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        # -- Validate --
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)

        # -- Early Stopping --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            log.info(f"  {name} early stop at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    log.info(f"  {name} training complete (best val_loss={best_val_loss:.4f})")
    return model


# ===========================================================================
# 6. ENSEMBLE PREDICTION
# ===========================================================================

def ensemble_predict(cnn_model, lstm_model, X):
    """Average voting ensemble from CNN1D and LSTM predictions.

    Returns:
        y_pred (np.ndarray): Binary predictions (0 or 1)
        y_proba (np.ndarray): Averaged probability scores
    """
    X_t = torch.FloatTensor(X).to(DEVICE)

    cnn_model.eval()
    lstm_model.eval()

    with torch.no_grad():
        cnn_proba = cnn_model(X_t.unsqueeze(1)).cpu().numpy()
        lstm_proba = lstm_model(X_t.unsqueeze(2)).cpu().numpy()

    # Average voting
    y_proba = (cnn_proba + lstm_proba) / 2.0
    y_pred = (y_proba >= 0.5).astype(int)

    return y_pred, y_proba


# ===========================================================================
# 7. STRATIFIED CROSS-VALIDATION
# ===========================================================================

def evaluate_cv(X, y):
    """5-fold Stratified Cross-Validation with full pipeline per fold.

    Returns:
        results (dict): Per-fold and mean metrics
    """
    log.info("=" * 60)
    log.info(f"{N_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION")
    log.info("=" * 60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_metrics = {
        'accuracy': [],
        'f1_score': [],
        'roc_auc': []
    }
    best_fold_auc = -1
    best_models = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        log.info(f"\n{'-' * 40}")
        log.info(f"FOLD {fold}/{N_FOLDS}")
        log.info(f"{'-' * 40}")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        log.info(f"Train: {X_train_fold.shape[0]} | Val: {X_val_fold.shape[0]}")

        # Balance training data (ADASYN + Mixup)
        X_train_bal, y_train_bal = balance_data(X_train_fold, y_train_fold)

        # Train models
        cnn_model, lstm_model = train_models(
            X_train_bal, y_train_bal,
            X_val_fold, y_val_fold
        )

        # Ensemble predict on validation set
        y_pred, y_proba = ensemble_predict(cnn_model, lstm_model, X_val_fold)

        # Compute metrics
        acc = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='weighted')
        try:
            auc = roc_auc_score(y_val_fold, y_proba)
        except ValueError:
            auc = 0.0

        fold_metrics['accuracy'].append(acc)
        fold_metrics['f1_score'].append(f1)
        fold_metrics['roc_auc'].append(auc)

        log.info(f"  Accuracy: {acc * 100:.2f}% | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

        # Track best fold
        if auc > best_fold_auc:
            best_fold_auc = auc
            best_models = (cnn_model.cpu().state_dict(), lstm_model.cpu().state_dict(),
                           cnn_model.__class__, lstm_model.__class__,
                           X_train_bal.shape[1])

    # -- Summary --
    log.info("\n" + "=" * 60)
    log.info("CROSS-VALIDATION RESULTS")
    log.info("=" * 60)

    results = {}
    print(f"\n{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 55)
    for metric_name, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        results[metric_name] = {'mean': mean_val, 'std': std_val, 'values': values}

        if metric_name == 'accuracy':
            print(f"{'Accuracy':<15} {mean_val * 100:>9.2f}% {std_val * 100:>9.2f}% {min_val * 100:>9.2f}% {max_val * 100:>9.2f}%")
        else:
            print(f"{metric_name:<15} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")

    results['best_models'] = best_models
    return results


# ===========================================================================
# 8. MAIN ORCHESTRATOR
# ===========================================================================

def main():
    """Full pipeline: preprocess -> feature select -> cross-validate -> save."""

    log.info("=" * 60)
    log.info("PARKINSON'S DISEASE DETECTION -- ADVANCED PIPELINE")
    log.info(f"Device: {DEVICE}")
    log.info("=" * 60)

    # 1. Preprocess
    X, y, scaler, feature_names = preprocess_data()

    # 2. Feature Selection (Boruta + PCA) -- runs once on full data
    X_reduced, boruta_mask, pca, selected_features = select_features(X, y, feature_names)

    # 3. Cross-Validation (balancing + training + evaluation per fold)
    results = evaluate_cv(X_reduced, y)

    # 4. Save artifacts
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Use glob to find the correct project directory with the special apostrophe
    project_matches = glob.glob('C:/Users/PREETI/Desktop/Parkinson*detection')
    if project_matches:
        project_dir = project_matches[0]

    save_dir = os.path.join(project_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)

    # Save pipeline components
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(boruta_mask, os.path.join(save_dir, 'boruta_mask.pkl'))
    joblib.dump(pca, os.path.join(save_dir, 'pca.pkl'))

    # Save best CNN1D and LSTM model weights
    if results.get('best_models'):
        cnn_state, lstm_state, cnn_cls, lstm_cls, input_dim = results['best_models']
        torch.save({
            'cnn_state_dict': cnn_state,
            'lstm_state_dict': lstm_state,
            'input_dim': input_dim,
        }, os.path.join(save_dir, 'ensemble_models.pt'))

    log.info(f"\n[OK] All artifacts saved to: {save_dir}")
    log.info("  - scaler.pkl          -- StandardScaler")
    log.info("  - boruta_mask.pkl     -- Boruta feature selection mask")
    log.info("  - pca.pkl             -- PCA transformer")
    log.info("  - ensemble_models.pt  -- CNN1D + LSTM weights")

    # Final summary
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Dataset:          756 samples x 753 features")
    print(f"  Boruta selected:  {np.sum(boruta_mask)} features")
    print(f"  PCA components:   {pca.n_components_}")
    print(f"  Ensemble:         CNN1D + LSTM (average voting)")
    print(f"  CV Folds:         {N_FOLDS}")
    print(f"  Mean Accuracy:    {results['accuracy']['mean'] * 100:.2f}% +/- {results['accuracy']['std'] * 100:.2f}%")
    print(f"  Mean F1-Score:    {results['f1_score']['mean']:.4f} +/- {results['f1_score']['std']:.4f}")
    print(f"  Mean ROC-AUC:     {results['roc_auc']['mean']:.4f} +/- {results['roc_auc']['std']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
