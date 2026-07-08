"""
app.py — Research Demonstration: Parkinson's Disease Detection
================================================================
CSV-upload inference using the same pipeline as training:
  scaler.pkl  →  boruta_mask.pkl  →  pca.pkl
  →  CNN1D + BiLSTM + RF + HistGBM + SVM
  →  AUC-weighted soft-voting  →  Optimal threshold
"""

import streamlit as st
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Disease — Research Demo",
    page_icon="🧠",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* ── metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 12px;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(99,102,241,0.25); }
.metric-value {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.82rem; color: #94a3b8; margin-top: 4px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
}

/* ── model prob mini-card ── */
.model-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.model-name { font-weight: 600; color: #c7d2fe; font-size: 0.9rem; }
.model-prob { font-weight: 700; font-size: 1.05rem; }
.prob-high  { color: #f87171; }
.prob-low   { color: #34d399; }

/* ── result banners ── */
.result-healthy {
    background: linear-gradient(135deg, #065f46, #047857);
    border-radius: 16px; padding: 28px; text-align: center;
    font-size: 1.5rem; font-weight: 700; color: #ecfdf5;
    border: 1px solid #34d399;
    animation: pulse-green 2.5s infinite;
}
.result-parkinsons {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border-radius: 16px; padding: 28px; text-align: center;
    font-size: 1.5rem; font-weight: 700; color: #fff1f2;
    border: 1px solid #f87171;
    animation: pulse-red 2.5s infinite;
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }
    50%      { box-shadow: 0 0 0 14px rgba(52,211,153,0); }
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(248,113,113,0.4); }
    50%      { box-shadow: 0 0 0 14px rgba(248,113,113,0); }
}

/* ── table ── */
.fold-table { width:100%; border-collapse:separate; border-spacing:0 6px; font-size:0.88rem; }
.fold-table th { color:#94a3b8; font-weight:600; text-transform:uppercase;
                 letter-spacing:0.06em; padding:6px 12px; font-size:0.75rem; }
.fold-table td { padding:8px 12px; background:rgba(255,255,255,0.05); color:#e2e8f0; }
.fold-table tr td:first-child { border-radius:8px 0 0 8px; }
.fold-table tr td:last-child  { border-radius:0 8px 8px 0; }

/* ── pipeline steps ── */
.pipeline-step {
    background: rgba(129,140,248,0.1);
    border-left: 3px solid #818cf8;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 4px 0;
    font-size: 0.88rem;
    color: #c7d2fe;
}

h1 { color: #e0e7ff !important; }
h2, h3 { color: #c7d2fe !important; }
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cpu")

# ── Non-feature columns that may appear in uploaded CSV ───────────────────────
DROP_COLS = {"id", "class", "ID", "Class", "label", "Label"}


# =============================================================================
# Resource loading (cached)
# =============================================================================

@st.cache_resource(show_spinner="🔧 Loading preprocessing pipeline…")
def load_preprocessing():
    """Load scaler → boruta_mask → pca saved by advanced_model.py."""
    errors = []
    artifacts = {}
    for name, fname in [("scaler", "scaler.pkl"),
                        ("boruta_mask", "boruta_mask.pkl"),
                        ("pca", "pca.pkl")]:
        path = os.path.join("models", fname)
        if os.path.exists(path):
            artifacts[name] = joblib.load(path)
        else:
            errors.append(f"❌ `models/{fname}` not found.")
    return artifacts, errors


@st.cache_resource(show_spinner="🤖 Loading ensemble models…")
def load_ensemble():
    """Load CNN1D + BiLSTM (from .pt) and RF + HistGBM + SVM (from .pkl)."""
    from src.advanced_model import CNN1D, LSTMModel
    errors = []

    # --- Deep learning ---
    pt_path = os.path.join("models", "ensemble_models.pt")
    cnn = lstm = None
    input_dim = weights = threshold = None

    if os.path.exists(pt_path):
        try:
            ck = torch.load(pt_path, map_location=DEVICE, weights_only=False)
            input_dim = ck["input_dim"]
            weights   = ck.get("weights")
            threshold = float(ck.get("threshold", 0.5))
            cnn  = CNN1D(input_dim).to(DEVICE)
            lstm = LSTMModel(input_dim).to(DEVICE)
            cnn.load_state_dict(ck["cnn_state_dict"])
            lstm.load_state_dict(ck["lstm_state_dict"])
            cnn.eval(); lstm.eval()
        except Exception as e:
            errors.append(f"❌ Failed to load ensemble_models.pt: {e}")
    else:
        errors.append("❌ `models/ensemble_models.pt` not found.")

    # --- Traditional ML ---
    ml = None
    ml_path = os.path.join("models", "ml_ensemble.pkl")
    if os.path.exists(ml_path):
        try:
            ml = joblib.load(ml_path)
        except Exception as e:
            errors.append(f"❌ Failed to load ml_ensemble.pkl: {e}")
    else:
        errors.append("❌ `models/ml_ensemble.pkl` not found.")

    return cnn, lstm, ml, input_dim, weights, threshold, errors


@st.cache_data(show_spinner=False)
def load_cv_results():
    p = os.path.join("models", "cv_results.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


# =============================================================================
# Inference
# =============================================================================

def preprocess(X_raw: np.ndarray, artifacts: dict):
    """Apply scaler → boruta mask → PCA exactly as during training."""
    X = artifacts["scaler"].transform(X_raw)
    X = X[:, artifacts["boruta_mask"]]
    X = artifacts["pca"].transform(X)
    return X


def run_full_ensemble(X_pca, cnn, lstm, ml, weights, threshold):
    """
    AUC-weighted soft voting across 5 models using saved weights & threshold.
    Returns: y_pred, y_prob, p_cnn, p_lstm, p_rf, p_hgb, p_svm
    """
    X_t = torch.FloatTensor(X_pca).to(DEVICE)

    with torch.no_grad():
        p_cnn  = cnn(X_t.unsqueeze(1)).cpu().numpy()
        p_lstm = lstm(X_t.unsqueeze(2)).cpu().numpy()

    p_rf  = ml["rf"].predict_proba(X_pca)[:, 1]
    p_hgb = ml["hgb"].predict_proba(X_pca)[:, 1]
    p_svm = ml["svm"].predict_proba(X_pca)[:, 1]

    probas = np.stack([p_cnn, p_lstm, p_rf, p_hgb, p_svm], axis=1)  # (n, 5)

    if weights is not None:
        w = np.array(weights)
    else:
        w = np.ones(5)
    w = w / w.sum()

    y_prob = (probas * w).sum(axis=1)
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob, p_cnn, p_lstm, p_rf, p_hgb, p_svm


# =============================================================================
# Load all resources at startup
# =============================================================================

prep_artifacts, prep_errors = load_preprocessing()
cnn_model, lstm_model, ml_models, model_input_dim, ens_weights, optimal_threshold, ens_errors = load_ensemble()
cv_results = load_cv_results()

all_errors = prep_errors + ens_errors
pipeline_ready = (
    len(prep_errors) == 0 and
    cnn_model is not None and
    lstm_model is not None and
    ml_models is not None
)

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## 🧠 Research Pipeline")

    st.markdown("""
    <div class="pipeline-step">📥 Upload 753-feature CSV</div>
    <div class="pipeline-step">⚖️ StandardScaler</div>
    <div class="pipeline-step">🔍 Boruta (131 features)</div>
    <div class="pipeline-step">🔢 PCA (99% variance → 69 dims)</div>
    <div class="pipeline-step">🤖 CNN1D · BiLSTM · RF · HistGBM · SVM</div>
    <div class="pipeline-step">🗳️ AUC-weighted soft voting</div>
    <div class="pipeline-step">🎯 Optimal threshold decision</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if pipeline_ready:
        st.success(f"✅ Pipeline ready  \n`input_dim = {model_input_dim}`  \n`threshold = {optimal_threshold:.4f}`")
    else:
        for e in all_errors:
            st.error(e)
        st.info("Run `python src/advanced_model.py` to train and save artifacts.")

    # ── CV Metrics ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 CV Metrics")

    if cv_results:
        acc = cv_results["accuracy"]
        f1  = cv_results["f1_score"]
        auc = cv_results["roc_auc"]
        pip = cv_results.get("pipeline", {})

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Accuracy", f"{acc['mean']*100:.2f}%", f"±{acc['std']*100:.2f}%")
            st.metric("F1-Score", f"{f1['mean']:.4f}",       f"±{f1['std']:.4f}")
        with c2:
            st.metric("ROC-AUC", f"{auc['mean']:.4f}",       f"±{auc['std']:.4f}")
            if pip:
                st.metric("PCA dims", pip.get("n_pca_components", "—"))

        with st.expander("📋 Per-Fold Breakdown"):
            rows = ""
            for i, (a, f, r) in enumerate(
                    zip(acc["per_fold"], f1["per_fold"], auc["per_fold"]), 1):
                rows += (f"<tr><td>Fold {i}</td><td>{a*100:.2f}%</td>"
                         f"<td>{f:.4f}</td><td>{r:.4f}</td></tr>")
            st.markdown(f"""
            <table class="fold-table">
              <tr><th>Fold</th><th>Accuracy</th><th>F1</th><th>AUC</th></tr>
              {rows}
            </table>""", unsafe_allow_html=True)

        if pip:
            st.caption(
                f"{pip.get('n_original_features','?')} features → "
                f"Boruta: {pip.get('n_boruta_features','?')} → "
                f"PCA: {pip.get('n_pca_components','?')}"
            )
    else:
        st.info("No metrics yet. Run `python src/advanced_model.py` first.")


# =============================================================================
# Main panel
# =============================================================================

st.title("🧠 Parkinson's Disease Detection — Research Demo")
st.markdown(
    "Upload a CSV containing **753 acoustic speech features** (same structure as "
    "`pd_speech_features.csv`). The app runs the **complete trained pipeline** — "
    "Scaler → Boruta → PCA → 5-model weighted ensemble — and returns predictions "
    "consistent with the **~92% cross-validated accuracy**."
)
st.markdown("---")

# ── Pipeline not ready guard ──────────────────────────────────────────────────
if not pipeline_ready:
    st.error("⚠️ One or more model artifacts are missing. See the sidebar for details.")
    st.code("python src/advanced_model.py", language="bash")
    st.stop()

# ── File uploader ─────────────────────────────────────────────────────────────
st.subheader("📂 Upload Patient Data")

col_up, col_info = st.columns([2, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "Upload a CSV file (753 feature columns, same format as training data)",
        type=["csv"],
        help="The CSV should have the same column structure as `pd_speech_features.csv`. "
             "The `id` and `class` columns are optional — they will be removed before inference.",
    )

with col_info:
    st.markdown("""
    <div class="metric-card" style="margin-top:0">
        <div class="metric-label">Expected Format</div>
        <div style="font-size:0.8rem;color:#94a3b8;margin-top:8px;text-align:left">
            • 753 feature columns<br>
            • Optional: <code>id</code>, <code>class</code><br>
            • One or more patient rows<br>
            • Same as training CSV
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Use training CSV as demo ──────────────────────────────────────────────────
use_demo = st.checkbox(
    "🔬 Use training data (`data/pd_speech_features.csv`) as demo input",
    value=False,
    help="Selects up to 10 random rows from the training dataset so you can verify the pipeline."
)

if use_demo and uploaded_file is None:
    demo_path = os.path.join("data", "pd_speech_features.csv")
    if os.path.exists(demo_path):
        demo_df = pd.read_csv(demo_path, header=1)
        demo_sample = demo_df.sample(min(10, len(demo_df)), random_state=42)
        st.info(f"ℹ️ Using {len(demo_sample)} random rows from the training CSV as demo input.")
        source_df = demo_sample
        has_labels = "class" in demo_sample.columns
    else:
        st.warning("`data/pd_speech_features.csv` not found. Upload a CSV file instead.")
        source_df = None
        has_labels = False
elif uploaded_file is not None:
    source_df = pd.read_csv(uploaded_file)
    has_labels = any(c in source_df.columns for c in ["class", "Class", "label", "Label"])
    st.success(f"✅ Loaded `{uploaded_file.name}` — {len(source_df)} row(s), {len(source_df.columns)} columns")
else:
    source_df = None
    has_labels = False

# =============================================================================
# Run inference
# =============================================================================

if source_df is not None:
    # ── Extract ground truth labels if present ────────────────────────────────
    y_true = None
    for lc in ["class", "Class", "label", "Label"]:
        if lc in source_df.columns:
            y_true = source_df[lc].values
            break

    # ── Drop non-feature columns ──────────────────────────────────────────────
    feature_df = source_df.drop(columns=[c for c in source_df.columns if c in DROP_COLS],
                                errors="ignore")

    # ── Validate feature count ────────────────────────────────────────────────
    n_features = prep_artifacts["scaler"].n_features_in_
    if feature_df.shape[1] != n_features:
        st.error(
            f"❌ Feature count mismatch: uploaded CSV has **{feature_df.shape[1]}** feature columns "
            f"but the trained scaler expects **{n_features}**. "
            "Please upload a CSV with the same structure as `pd_speech_features.csv`."
        )
        st.stop()

    X_raw = feature_df.values.astype(np.float64)

    # ── Apply preprocessing pipeline ─────────────────────────────────────────
    with st.spinner("⚙️ Applying preprocessing pipeline (Scaler → Boruta → PCA)…"):
        try:
            X_pca = preprocess(X_raw, prep_artifacts)
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {e}")
            st.stop()

    # ── Run 5-model ensemble ──────────────────────────────────────────────────
    with st.spinner("🤖 Running 5-model AUC-weighted ensemble…"):
        try:
            y_pred, y_prob, p_cnn, p_lstm, p_rf, p_hgb, p_svm = run_full_ensemble(
                X_pca, cnn_model, lstm_model, ml_models, ens_weights, optimal_threshold
            )
        except Exception as e:
            st.error(f"❌ Inference failed: {e}")
            st.stop()

    n_samples = len(y_pred)

    st.markdown("---")

    # =========================================================================
    # Results — single sample
    # =========================================================================
    if n_samples == 1:
        st.subheader("📊 Prediction Result")

        label = y_pred[0]
        prob  = float(y_prob[0])

        if label == 0:
            st.markdown(
                '<div class="result-healthy">✅ No Parkinson\'s Indicators Detected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-parkinsons">⚠️ Parkinson\'s Indicators Detected</div>',
                unsafe_allow_html=True,
            )

        if y_true is not None:
            gt = int(y_true[0])
            correct = (gt == label)
            gt_label = "Parkinson's" if gt == 1 else "Healthy"
            result_label = "✅ Correct" if correct else "❌ Incorrect"
            st.markdown(f"**Ground Truth:** {gt_label}  {result_label}")

        st.markdown("<br>", unsafe_allow_html=True)

        risk_pct    = prob * 100
        healthy_pct = (1 - prob) * 100
        confidence  = max(prob, 1 - prob) * 100

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{risk_pct:.1f}%</div>
                <div class="metric-label">Parkinson's Risk</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{healthy_pct:.1f}%</div>
                <div class="metric-label">Healthy Probability</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{confidence:.1f}%</div>
                <div class="metric-label">Model Confidence</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>**Ensemble Risk Score**", unsafe_allow_html=True)
        st.progress(float(prob))
        st.caption(f"Threshold: **{optimal_threshold:.4f}** | "
                   f"Ensemble probability: **{prob*100:.2f}%**")

        # ── Individual model breakdown ────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Individual Model Breakdown")

        model_probs = {
            "CNN1D":          float(p_cnn[0]),
            "BiLSTM-Attention": float(p_lstm[0]),
            "Random Forest":  float(p_rf[0]),
            "HistGradientBoosting": float(p_hgb[0]),
            "Calibrated SVM": float(p_svm[0]),
        }

        if ens_weights:
            w_norm = np.array(ens_weights) / np.sum(ens_weights)
            weight_map = dict(zip(
                ["CNN1D", "BiLSTM-Attention", "Random Forest", "HistGradientBoosting", "Calibrated SVM"],
                w_norm
            ))
        else:
            weight_map = {k: 0.2 for k in model_probs}

        for mname, mp in model_probs.items():
            cls = "prob-high" if mp >= optimal_threshold else "prob-low"
            pred_label = "⚠️ PD" if mp >= optimal_threshold else "✅ Healthy"
            wt = weight_map.get(mname, 0.2)
            st.markdown(f"""
            <div class="model-card">
                <div>
                    <div class="model-name">{mname}</div>
                    <div style="font-size:0.75rem;color:#64748b;">weight: {wt:.3f}</div>
                </div>
                <div>
                    <span class="model-prob {cls}">{mp*100:.1f}%</span>
                    <span style="margin-left:12px;font-size:0.8rem;color:#94a3b8;">{pred_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # Results — multiple samples
    # =========================================================================
    else:
        st.subheader(f"📊 Predictions — {n_samples} Samples")

        # ── Summary metrics ───────────────────────────────────────────────────
        n_pd      = int(np.sum(y_pred == 1))
        n_healthy = int(np.sum(y_pred == 0))
        avg_risk  = float(np.mean(y_prob)) * 100

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{n_pd}</div>
                <div class="metric-label">Parkinson's Detected</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{n_healthy}</div>
                <div class="metric-label">Healthy Predicted</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{avg_risk:.1f}%</div>
                <div class="metric-label">Mean Risk Score</div>
            </div>""", unsafe_allow_html=True)

        # ── Accuracy if labels available ──────────────────────────────────────
        if y_true is not None:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except Exception:
                auc = float("nan")

            st.markdown("#### 📈 Performance Against Ground Truth")
            pa1, pa2, pa3 = st.columns(3)
            with pa1:
                st.metric("Accuracy", f"{acc*100:.2f}%")
            with pa2:
                st.metric("F1-Score", f"{f1:.4f}")
            with pa3:
                st.metric("ROC-AUC", f"{auc:.4f}" if not np.isnan(auc) else "N/A")

        # ── Per-sample results table ──────────────────────────────────────────
        results_data = {
            "Sample": range(1, n_samples + 1),
            "Prediction":  ["Parkinson's" if p == 1 else "Healthy" for p in y_pred],
            "Ensemble %":  [f"{p*100:.2f}%" for p in y_prob],
            "CNN1D %":     [f"{p*100:.2f}%" for p in p_cnn],
            "BiLSTM %":    [f"{p*100:.2f}%" for p in p_lstm],
            "RF %":        [f"{p*100:.2f}%" for p in p_rf],
            "HistGBM %":   [f"{p*100:.2f}%" for p in p_hgb],
            "SVM %":       [f"{p*100:.2f}%" for p in p_svm],
        }
        if y_true is not None:
            results_data["Ground Truth"] = ["Parkinson's" if g == 1 else "Healthy" for g in y_true]
            results_data["Correct"] = ["✅" if p == g else "❌" for p, g in zip(y_pred, y_true)]

        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # ── Download predictions ──────────────────────────────────────────────
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv_bytes,
            file_name="parkinson_predictions.csv",
            mime="text/csv",
        )

    # ── Methodology note ──────────────────────────────────────────────────────
    st.markdown("---")
    st.info(
        "ℹ️ **Methodology:** Predictions use the identical preprocessing pipeline applied during "
        "training — StandardScaler → Boruta feature selection (131 features) → PCA (99% variance, "
        f"{prep_artifacts['pca'].n_components_} components) — followed by the 5-model "
        "AUC-weighted ensemble trained on the UCI Parkinson's Speech Dataset "
        "(Sakar et al., 2018, 756 samples, 753 features). "
        "Reported cross-validated accuracy: **~92%**."
    )
    st.warning(
        "⚠️ **Disclaimer:** This is a research demonstration tool, **not** a clinical diagnostic "
        "system. All predictions are for research purposes only. Always consult a qualified neurologist."
    )