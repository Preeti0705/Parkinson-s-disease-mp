# Methodology Audit — Parkinson's Disease Detection Pipeline

Comparing the architectural diagram against the actual implementation across all source files.

---

## ✅ Overall Verdict: **The diagram is ACCURATE — the code faithfully implements every stage shown**

Every box in the diagram has a direct, traceable implementation in the source code. Below is a phase-by-phase breakdown.

---

## Phase 1 — Input Phase

| Diagram Says | Code Does | Status |
|---|---|---|
| Raw Patient Acoustic Data (753 Voice Features) | `preprocess_data()` loads `pd_speech_features.csv`; `cv_results.json` confirms `n_original_features: 753` | ✅ Match |

**Evidence:** `advanced_model.py` L196–L202 — drops `id` and `class` columns, leaving 753 feature columns.

---

## Phase 2 — Data Preprocessing Phase

### 2a. StandardScaler (Zero-mean, unit-variance normalization)

| Diagram Says | Code Does | Status |
|---|---|---|
| StandardScaler | `StandardScaler().fit_transform(X)` in `preprocess_data()` | ✅ Match |

**Evidence:** `advanced_model.py` L205–L206
```python
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
Artifact saved as `models/scaler.pkl`. The **same** scaler is applied during inference in `app.py` and `evaluate_saved_models.py`.

---

### 2b. Boruta Feature Selection (Removes statistical noise)

| Diagram Says | Code Does | Status |
|---|---|---|
| Boruta Feature Selection | `BorutaPy` wrapping `RandomForestClassifier` in `select_features()` | ✅ Match |
| Reduces noise | 753 → 131 features (confirmed in `cv_results.json`) | ✅ Match |

**Evidence:** `advanced_model.py` L221–L228
```python
rf_b   = RandomForestClassifier(n_estimators=100, max_depth=7, ...)
boruta = BorutaPy(estimator=rf_b, n_estimators='auto', max_iter=100, ...)
boruta.fit(X, y)
```
`cv_results.json` confirms: `"n_boruta_features": 131`

---

### 2c. PCA (Reduces dimensionality while retaining 99% variance)

| Diagram Says | Code Does | Status |
|---|---|---|
| PCA (99% variance) | `PCA(n_components=0.99)` | ✅ Match |
| Dimensionality reduction | 131 → 69 PCA components (confirmed in `cv_results.json`) | ✅ Match |

**Evidence:** `advanced_model.py` L233–L236
```python
pca   = PCA(n_components=PCA_VARIANCE, random_state=SEED)  # PCA_VARIANCE = 0.99
X_red = pca.fit_transform(X_sel)
```
`cv_results.json` confirms: `"n_pca_components": 69`

---

## Phase 3 — Multi-Model Inference Phase

### 3a. Residual CNN1D (Deep Learning)

| Diagram Says | Code Does | Status |
|---|---|---|
| Residual CNN1D | `CNN1D` class with `_ResBlock` residual connections | ✅ Match |

**Evidence:** `advanced_model.py` L99–L148 — `_ResBlock` uses two Conv1D layers with a skip connection:
```python
return self.relu(x + self.body(x))
```

---

### 3b. BiLSTM + Self-Attention (Deep Learning)

| Diagram Says | Code Does | Status |
|---|---|---|
| BiLSTM + Self-Attention | `LSTMModel` with `bidirectional=True` + attention scoring | ✅ Match |

**Evidence:** `advanced_model.py` L151–L177
```python
self.lstm1 = nn.LSTM(input_size=1,   hidden_size=64, bidirectional=True)
self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True)
self.attn  = nn.Linear(128, 1)           # self-attention scoring
w = torch.softmax(self.attn(x), dim=1)  # attention weights
x = (w * x).sum(dim=1)                  # attention pooling
```

---

### 3c. Random Forest (Traditional ML)

| Diagram Says | Code Does | Status |
|---|---|---|
| Random Forest | `RandomForestClassifier(n_estimators=500, class_weight='balanced')` | ✅ Match |

**Evidence:** `advanced_model.py` L336–L341

---

### 3d. HistGradientBoosting (Traditional ML)

| Diagram Says | Code Does | Status |
|---|---|---|
| HistGradientBoosting | `HistGradientBoostingClassifier(max_iter=500, class_weight='balanced')` | ✅ Match |

**Evidence:** `advanced_model.py` L344–L349

---

### 3e. Calibrated SVM (Traditional ML)

| Diagram Says | Code Does | Status |
|---|---|---|
| Calibrated SVM | `CalibratedClassifierCV(SVC(kernel='rbf'), cv=3)` | ✅ Match |

**Evidence:** `advanced_model.py` L352–L354
```python
svm_base = SVC(kernel='rbf', class_weight='balanced', probability=True)
svm      = CalibratedClassifierCV(svm_base, cv=3)
```

---

## Phase 4 — Ensemble & Decision Phase

### 4a. AUC-Weighted Soft Voting

| Diagram Says | Code Does | Status |
|---|---|---|
| Models weighted by historical validation AUC | AUC scores clipped and normalized into voting weights | ✅ Match |

**Evidence:** `advanced_model.py` L454–L458
```python
aucs    = np.array([safe_auc(p) for p in [p_cnn, p_lstm, p_rf, p_hgb, p_svm]])
weights = np.clip(aucs, 0.5, 1.0) - 0.5   # shift: 0.5 AUC → weight 0
weights = weights / weights.sum()
y_pred, y_prob = ensemble_predict(..., weights=weights)
```

The 5 model probabilities are stacked and matrix-multiplied by normalized AUC weights — exactly the soft voting described in the diagram.

---

### 4b. Youden's J Statistic (Optimal dynamic threshold)

| Diagram Says | Code Does | Status |
|---|---|---|
| Youden's J Statistic (optimal threshold for imbalanced data) | `find_optimal_threshold()` maximizes `TPR - FPR` on the ROC curve | ✅ Match |

**Evidence:** `advanced_model.py` L390–L394
```python
def find_optimal_threshold(y_true, y_proba):
    """Youden's J statistic: maximize TPR - FPR."""
    fpr, tpr, thresh = roc_curve(y_true, y_proba)
    j = tpr - fpr
    return float(thresh[np.argmax(j)])
```
Saved threshold in `cv_results.json`: **0.5573** (not a naive 0.5, confirming it's dynamically computed).

---

## Phase 5 — Output Phase

### 5a. Final Prediction (Parkinson's vs. Healthy)

| Diagram Says | Code Does | Status |
|---|---|---|
| Binary classification output | `y_pred = (y_prob >= threshold).astype(int)` then mapped to label strings | ✅ Match |

**Evidence:** `app.py` L459–L468 renders `"✅ No Parkinson's Indicators"` or `"⚠️ Parkinson's Indicators"` banners.

---

### 5b. Confidence Breakdown & Risk Score Dashboard

| Diagram Says | Code Does | Status |
|---|---|---|
| Confidence Breakdown | Per-model probability cards with voting weights displayed | ✅ Match |
| Risk Score Dashboard | Parkinson's Risk %, Healthy %, Model Confidence %, progress bar | ✅ Match |

**Evidence:** `app.py` L479–L541 — Three metric cards (Risk %, Healthy %, Confidence %), progress bar, and individual model breakdown cards with weights.

---

## Additional Implementation Details NOT in the Diagram (but correctly implemented)

| Component | File / Lines | Notes |
|---|---|---|
| **ADASYN oversampling** | `balance_data()`, L245–L254 | Applied **only per training fold**, never to val/test — methodologically correct |
| **Focal Loss** | `FocalLoss` class, L75–L92 | Used for DL training to handle class imbalance |
| **5-Fold Stratified CV** | `evaluate_cv()`, L401–L523 | Proper stratification maintained across folds |
| **Early stopping + Cosine LR** | `_train_single_dl()`, L261–L299 | Patience=25, `CosineAnnealingLR` scheduler |
| **Gradient clipping** | L279 | `clip_grad_norm_(model.parameters(), 1.0)` |

---

## ⚠️ One Methodological Concern to Be Aware Of

> **Warning — Mild Preprocessing Data Leakage:**
> Boruta + PCA are fit on the FULL dataset before cross-validation begins.

In `advanced_model.py` `main()`:
```python
X, y, scaler, feature_names = preprocess_data()               # scaler fit on ALL data
X_red, boruta_mask, pca, sel_names = select_features(X, y, feature_names)  # Boruta + PCA fit on ALL data
results = evaluate_cv(X_red, y)                               # CV runs on already-transformed data
```

This means Boruta and PCA have "seen" the validation samples before each fold's evaluation — a mild form of **preprocessing data leakage** that can slightly inflate reported CV accuracy.

The ADASYN balancing inside the loop is correctly done **per fold**, which is good. For a fully rigorous implementation, Boruta and PCA should also be re-fit inside each fold on the training split only.

This does **not** invalidate the project — it is an extremely common simplification in academic work — but it is worth acknowledging for any formal report.

---

## Summary Table

| Diagram Component | Implemented? | File | Lines |
|---|---|---|---|
| 753 Voice Features Input | ✅ Yes | `advanced_model.py` | L196–202 |
| StandardScaler | ✅ Yes | `advanced_model.py` | L205–206 |
| Boruta Feature Selection | ✅ Yes | `advanced_model.py` | L221–228 |
| PCA (99% variance) | ✅ Yes | `advanced_model.py` | L233–236 |
| Residual CNN1D | ✅ Yes | `advanced_model.py` | L99–148 |
| BiLSTM + Self-Attention | ✅ Yes | `advanced_model.py` | L151–177 |
| Random Forest | ✅ Yes | `advanced_model.py` | L336–341 |
| HistGradientBoosting | ✅ Yes | `advanced_model.py` | L344–349 |
| Calibrated SVM | ✅ Yes | `advanced_model.py` | L352–354 |
| AUC-Weighted Soft Voting | ✅ Yes | `advanced_model.py` | L454–458 |
| Youden's J Threshold | ✅ Yes | `advanced_model.py` | L390–394 |
| Final Binary Prediction | ✅ Yes | `app.py` | L459–468 |
| Risk Score Dashboard | ✅ Yes | `app.py` | L479–541 |

---

**Achieved Accuracy:** 91.8% ± 2.5% &nbsp;|&nbsp; **ROC-AUC:** 96.4% ± 2.3% &nbsp;|&nbsp; Evaluation: 5-Fold Stratified Cross-Validation
