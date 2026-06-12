import streamlit as st
import tempfile
import os
import glob
import json
import warnings
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎙️ Parkinson's Voice Analysis",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
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

.fold-table { width:100%; border-collapse:separate; border-spacing:0 6px; font-size:0.88rem; }
.fold-table th { color:#94a3b8; font-weight:600; text-transform:uppercase;
                 letter-spacing:0.06em; padding:6px 12px; font-size:0.75rem; }
.fold-table td { padding:8px 12px; background:rgba(255,255,255,0.05); color:#e2e8f0; }
.fold-table tr td:first-child { border-radius:8px 0 0 8px; }
.fold-table tr td:last-child  { border-radius:0 8px 8px 0; }

h1 { color:#e0e7ff !important; }
h2, h3 { color:#c7d2fe !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — all inside Streamlit execution context via @st.cache_resource
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cpu')

# Column names in pd_speech_features.csv that correspond to audio_prepro features
# (discovered by inspecting the CSV header)
AUDIO_FEATURE_COLS = [
    'locPctJitter', 'locAbsJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
    'locShimmer', 'locDbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer',
    'NHR', 'HNR',
]


@st.cache_resource(show_spinner="🔧 Loading CNN1D + BiLSTM ensemble…")
def load_ensemble():
    """Load Residual-CNN1D + BiLSTM-Attention models from models/ensemble_models.pt."""
    from src.inference_models import CNN1D, LSTMModel
    pt_path = os.path.join('models', 'ensemble_models.pt')
    if not os.path.exists(pt_path):
        return None, None, None, 0.5, "❌ `models/ensemble_models.pt` not found. Run `python src/advanced_model.py` first."
    try:
        ck = torch.load(pt_path, map_location=DEVICE, weights_only=False)
        input_dim = ck['input_dim']
        threshold = float(ck.get('threshold', 0.5))   # optimal threshold saved by training
        cnn  = CNN1D(input_dim).to(DEVICE)
        lstm = LSTMModel(input_dim).to(DEVICE)
        cnn.load_state_dict(ck['cnn_state_dict'])
        lstm.load_state_dict(ck['lstm_state_dict'])
        cnn.eval(); lstm.eval()
        return cnn, lstm, input_dim, threshold, None
    except Exception as e:
        return None, None, None, 0.5, f"❌ Failed to load ensemble: {e}"


@st.cache_resource(show_spinner="📐 Building audio feature scaler from training data…")
def load_audio_scaler():
    """
    Fit a StandardScaler on the 13 matching acoustic columns from the training CSV.
    Returns (scaler, error_msg).
    """
    try:
        import pandas as pd
        csv_matches = glob.glob('C:/Users/PREETI/Desktop/Parkinson*/data/pd_speech_features.csv')
        csv_path = csv_matches[0] if csv_matches else os.path.join('data', 'pd_speech_features.csv')
        df = pd.read_csv(csv_path, header=1)

        # Use known matching column names; fall back to first 13 feature cols
        available = [c for c in df.columns if c not in ['id', 'class']]
        cols_to_use = [c for c in AUDIO_FEATURE_COLS if c in available]
        if len(cols_to_use) < 13:
            cols_to_use = available[:13]

        X = df[cols_to_use].values.astype(np.float32)
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler, cols_to_use, None
    except Exception as e:
        return None, [], f"⚠️ Could not build audio scaler: {e}"


@st.cache_data(show_spinner=False)
def load_cv_results():
    """Load cross-validation results JSON produced by advanced_model.py."""
    p = os.path.join('models', 'cv_results.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def extract_audio_features(audio_bytes):
    """Save audio bytes to temp file, run extract_features, return (array, error)."""
    tmp = None
    try:
        from src.audio_prepro import extract_features
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        with open(tmp, 'wb') as f:
            f.write(audio_bytes)
        feats = extract_features(tmp)
        return feats, None
    except Exception as e:
        return None, str(e)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def run_ensemble(feats_raw, scaler, cnn_model, lstm_model, threshold=0.5):
    """
    Scale 13 audio features, run Residual-CNN1D + BiLSTM-Attention ensemble.
    Uses the optimal Youden-J threshold saved during training.
    Returns (label, probability, cnn_p, lstm_p).
    """
    if scaler is not None:
        fs = scaler.transform(feats_raw.reshape(1, -1)).astype(np.float32)
    else:
        fs = feats_raw.reshape(1, -1).astype(np.float32)

    X_t = torch.from_numpy(fs).to(DEVICE)                          # (1, 13)
    with torch.no_grad():
        cnn_p  = float(cnn_model(X_t.unsqueeze(1)).cpu())           # CNN  : (1,1,13)
        lstm_p = float(lstm_model(X_t.unsqueeze(2)).cpu())          # LSTM : (1,13,1)
    prob  = (cnn_p + lstm_p) / 2.0
    label = 1 if prob >= threshold else 0
    return label, prob, cnn_p, lstm_p


# ═════════════════════════════════════════════════════════════════════════════
# Load resources (inside Streamlit execution)
# ═════════════════════════════════════════════════════════════════════════════

cnn_model, lstm_model, model_input_dim, optimal_threshold, ensemble_err = load_ensemble()
audio_scaler, scaler_cols, scaler_err                                    = load_audio_scaler()
cv_results                                                               = load_cv_results()

# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📋 Patient Profile")
    age       = st.slider("Age", 18, 100, 60)
    sex_label = st.selectbox("Sex", ["Male", "Female"])

    st.markdown("---")
    st.markdown("## 🤖 Model")
    st.markdown("""
    **Hybrid Ensemble v2**  
    🔵 Residual-CNN1D + BiLSTM-Attention  
    🟢 Random Forest + HistGBM + SVM (RBF)  
    📐 Boruta → PCA (99% var)  
    ⚖️ Focal Loss · ADASYN · AUC-weighted voting  
    📊 5-Fold Stratified CV
    """)


    if ensemble_err:
        st.error(ensemble_err)
    else:
        st.success(f"✅ Ensemble loaded (input_dim = {model_input_dim})")

    # ── CV Metrics ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 CV Metrics")

    if cv_results:
        acc = cv_results['accuracy']
        f1  = cv_results['f1_score']
        auc = cv_results['roc_auc']
        pip = cv_results.get('pipeline', {})

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Accuracy", f"{acc['mean']*100:.2f}%", f"±{acc['std']*100:.2f}%")
            st.metric("F1-Score", f"{f1['mean']:.4f}",       f"±{f1['std']:.4f}")
        with c2:
            st.metric("ROC-AUC", f"{auc['mean']:.4f}",       f"±{auc['std']:.4f}")
            if pip:
                st.metric("PCA dims", pip.get('n_pca_components', '—'))

        with st.expander("📋 Per-Fold Breakdown"):
            rows = ""
            for i, (a, f, r) in enumerate(zip(acc['per_fold'], f1['per_fold'], auc['per_fold']), 1):
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
        st.info(
            "No metrics found yet.\n\n"
            "Run `python src/advanced_model.py` to train and save metrics."
        )

# ═════════════════════════════════════════════════════════════════════════════
# Main panel
# ═════════════════════════════════════════════════════════════════════════════

st.title("🎙️ Parkinson's Disease Voice Analysis")
st.markdown(
    "Record a short sustained **\"ah\"** sound. The **CNN1D + LSTM ensemble** "
    "will analyse your voice and estimate the probability of Parkinson's disease indicators."
)
st.markdown("---")

if ensemble_err:
    st.error(ensemble_err)
    st.info("Train the model first by running: `python src/advanced_model.py`")
    st.stop()

st.subheader("🎤 Record Your Voice")
from st_audiorec import st_audiorec
audio_bytes = st_audiorec()

if audio_bytes is not None:
    st.info("✅ Recording captured — running ensemble inference…")

    with st.spinner("🧠 Running Hybrid Ensemble (CNN1D + BiLSTM + RF + HistGBM + SVM)…"):
        feats, feat_err = extract_audio_features(audio_bytes)

    if feat_err:
        st.error(f"Feature extraction failed: {feat_err}")
        st.stop()

    label, prob, cnn_p, lstm_p = run_ensemble(
        feats, audio_scaler, cnn_model, lstm_model, threshold=optimal_threshold
    )

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    # ── Result banner ────────────────────────────────────────────────────────
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

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3 summary metric cards ───────────────────────────────────────────────
    risk_pct, healthy_pct = prob * 100, (1 - prob) * 100
    confidence = max(prob, 1 - prob) * 100

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

    # ── Risk progress bar ────────────────────────────────────────────────────
    st.markdown("<br>**Parkinson's Risk Score**", unsafe_allow_html=True)
    st.progress(float(prob))
    st.caption(f"Ensemble score = (CNN1D {cnn_p*100:.1f}% + LSTM {lstm_p*100:.1f}%) ÷ 2 = **{prob*100:.1f}%**")

    # ── CNN vs LSTM breakdown cards ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Model Breakdown")
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{cnn_p*100:.1f}%</div>
            <div class="metric-label">CNN1D Risk Score</div>
        </div>""", unsafe_allow_html=True)
    with bc2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{lstm_p*100:.1f}%</div>
            <div class="metric-label">LSTM Risk Score</div>
        </div>""", unsafe_allow_html=True)

    # ── Feature values table ─────────────────────────────────────────────────
    with st.expander("🔬 Extracted Voice Features"):
        import pandas as pd
        feat_names = [
            'Jitter (local)', 'Jitter (abs)', 'Jitter (RAP)', 'Jitter (PPQ5)', 'Jitter (DDP)',
            'Shimmer (local)', 'Shimmer (dB)', 'Shimmer (APQ3)', 'Shimmer (APQ5)',
            'Shimmer (APQ11)', 'Shimmer (DDA)', 'NHR', 'HNR'
        ]
        feat_df = pd.DataFrame({'Feature': feat_names, 'Value': [f"{v:.6f}" for v in feats]})
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.info(
        "ℹ️ **About the model**: Trained on the UCI Parkinson's Speech Dataset "
        "(756 samples, 753 features) with ADASYN + Mixup augmentation, "
        "Boruta feature selection, PCA, and 5-fold stratified CV."
    )
    st.warning(
        "⚠️ **Disclaimer**: This is a research screening tool, **not** a medical diagnosis. "
        "Always consult a qualified neurologist."
    )
    st.markdown("**Review your recording:**")
    st.audio(audio_bytes, format='audio/wav')