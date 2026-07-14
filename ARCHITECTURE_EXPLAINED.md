# 🧠 Parkinson's Disease Detection — Complete Architecture Explained From Scratch

> This document explains every single component of this project in plain English, with simple definitions for every technical term.

---

## 📌 Table of Contents
1. [What Are We Actually Trying to Do?](#1-what-are-we-actually-trying-to-do)
2. [The Dataset](#2-the-dataset-the-raw-material)
3. [The Pipeline Overview](#3-the-pipeline-step-by-step)
4. [Preprocessing](#4-preprocessing-cleaning-the-data)
5. [The 5 Models](#5-the-5-models)
6. [The Ensemble](#6-the-ensemble-combining-all-5-models)
7. [Youden's J Threshold](#7-youdens-j-threshold)
8. [Cross-Validation](#8-cross-validation-how-we-measure-performance-fairly)
9. [Loss Functions](#9-loss-functions-how-models-learn)
10. [Saved Artifacts](#10-what-gets-saved-the-artifacts)
11. [The Web App](#11-the-web-app-apppy)
12. [The Complete Picture](#12-the-complete-picture-everything-together)
13. [Quick Glossary](#13-quick-glossary-all-terms-in-one-place)

---

## 1. What Are We Actually Trying to Do?

**The Real-World Goal:**
Parkinson's Disease is a brain disorder that slowly damages nerve cells. One of its earliest and most measurable signs is changes in a person's **voice**. People with Parkinson's often have:
- A shaky, trembling voice
- Inconsistent loudness
- A "breathy" or hoarse quality

The idea: if we record someone saying **"ahhhhh"** and measure enough properties of that sound, a computer can learn to tell whether that person likely has Parkinson's — just from their voice.

**This project builds that computer system.**

---

## 2. The Dataset (The Raw Material)

**File:** `data/pd_speech_features.csv`

Think of this as a giant Excel spreadsheet with:
- **756 rows** → 756 voice recordings (from 252 patients, each recorded 3 times)
- **754 columns** → 753 measured voice properties + 1 label column (`class`)
- **Label:** `0` = Healthy person, `1` = Has Parkinson's

### What are these 753 features?

Each feature is a number that describes one specific property of the voice recording:

| Feature Type | Simple Definition |
|---|---|
| **Jitter** | How much does the pitch wobble? Parkinson's voices wobble more |
| **Shimmer** | How much does the loudness wobble? |
| **HNR** | Ratio of clear voice vs. noise/breathiness |
| **MFCC** | How the voice sounds across different frequency bands (like a detailed audio fingerprint) |
| **Wavelet** | Captures both time and frequency info simultaneously |
| **RPDE / DFA / PPE** | Mathematical measures of how "chaotic" or "irregular" the voice is |

### The Class Imbalance Problem

> **Simple Definition — Class Imbalance:** When one group has way more samples than another. Like having 750 cats and only 250 dogs in a dataset for a cat/dog classifier.

Our dataset:
- **564 samples** → Parkinson's (75%)
- **192 samples** → Healthy (25%)

This is a problem! If the model just predicts "Parkinson's" for every single person, it gets **75% accuracy** without learning anything useful. We solve this with special techniques (explained later).

---

## 3. The Pipeline (Step-by-Step)

Think of the pipeline like an assembly line in a factory. Raw data goes in one end, and a trained, ready-to-use AI model comes out the other end.

```
Raw CSV Data (756 people × 753 features)
        ↓
  STEP 1: StandardScaler    (normalize all numbers to the same scale)
        ↓
  STEP 2: Boruta            (keep only the important features)
        ↓
  STEP 3: PCA               (compress further without losing info)
        ↓
  STEP 4: ADASYN            (fix the class imbalance)
        ↓
  STEP 5: Train 5 Models    (simultaneously, in each CV fold)
        ↓
  STEP 6: Ensemble          (combine all 5 model opinions)
        ↓
  STEP 7: Youden's J        (find the best decision threshold)
        ↓
  STEP 8: Save to disk      (all models + preprocessing artifacts)
        ↓
  STEP 9: Web App           (use saved artifacts for new predictions)
```

---

## 4. Preprocessing (Cleaning the Data)

### Step 1 — StandardScaler

> **Simple Definition:** Puts all numbers on the same scale so no feature unfairly dominates.

**The Problem:** Some features might be in the range `0.001–0.1`, while others are `1000–50000`. When one number is 1000× bigger than another, the model might think it's more "important" just because of its size — not because it actually is.

**The Fix:** StandardScaler transforms every feature so it has:
- **Mean = 0** (the average becomes zero)
- **Standard Deviation = 1** (the spread is normalized)

**Think of it like:** Converting all temperatures to Celsius so you can compare them fairly.

```
Before scaling:  Feature A = 0.003,   Feature B = 45000
After scaling:   Feature A = -0.8,    Feature B =  1.2
```

Both are now on the same scale. No feature unfairly dominates.

---

### Step 2 — Boruta Feature Selection

> **Simple Definition:** An algorithm that figures out which of the 753 features actually matter and which ones are just noise.

**The Problem:** With 753 features, many of them might be:
- Redundant (two features measuring nearly the same thing)
- Irrelevant noise that confuses the model
- Computationally expensive to process

**The Fix — Boruta Algorithm:**
1. Creates "shadow" (fake, shuffled) copies of every feature
2. Trains a Random Forest on both real + fake features
3. Any real feature that performs **worse than its own fake copy** is declared useless and removed
4. Repeats this many times to be statistically certain

**Result:** 753 features → **131 features** (the genuinely important ones)

**Think of it like:** You have 753 job candidates. Boruta is the hiring manager who tests them all and keeps only the 131 who actually pass the test.

---

### Step 3 — PCA (Principal Component Analysis)

> **Simple Definition:** Compresses information by finding the directions in data that explain the most variation.

**The Problem:** Even after Boruta, 131 features still contain a lot of redundancy (e.g., 5 different shimmer measurements are all very similar to each other).

**The Fix — PCA:**
1. Finds the mathematical "directions" (called **principal components**) along which data varies the most
2. Keeps only the directions needed to explain **99% of the variance** (information)
3. Projects all data onto these fewer, uncorrelated directions

**Result:** 131 features → **69 components** (while keeping 99% of all the information)

**Think of it like:** You have 131 slightly different photos of the same person from slightly different angles. PCA finds the 69 most meaningful "viewpoints" that together capture 99% of what makes that person unique. The other 62 viewpoints are redundant.

---

### Step 4 — ADASYN (Adaptive Synthetic Sampling)

> **Simple Definition:** Creates artificial "Healthy" patient data to balance the dataset during training.

**The Problem:** Remember — only 25% of samples are "Healthy". If we don't fix this, the model becomes biased toward predicting Parkinson's.

**The Fix — ADASYN:**
1. Looks at existing Healthy samples
2. Finds their nearest neighbors in feature space
3. Creates new **synthetic** (artificial but mathematically realistic) Healthy samples between them

**Result:** Training set grows from ~604 samples to ~876 samples, now much more balanced.

> ⚠️ **Critical Rule:** ADASYN is applied **only to training data** per fold. The test/validation data always contains only real patients — never synthetic ones.

**Think of it like:** Your dataset has 75 dogs and 25 cats. ADASYN creates 50 more realistic-looking cat examples to make it 75/75 fair.

---

## 5. The 5 Models

Instead of using just one AI model, we train **5 different models** and let them vote together. Each model is good at different things.

---

### Model 1 — Residual CNN1D (Deep Learning)

> **Simple Definition:** A neural network that scans across the 69 features like reading a sentence, detecting local patterns.

**CNN = Convolutional Neural Network**
- A "filter" slides along the feature vector, detecting patterns in **local groups of features**
- Like how your eye notices a curve or line in part of an image

**Residual = Skip Connection**
- The input to a layer is **added directly to its output**: `output = layer(input) + input`
- This prevents the network from "forgetting" earlier information
- Think of it as a **shortcut highway** that carries information across multiple layers

**Architecture in this project:**
```
69 PCA components (treated as a 1D signal of length 69)
        ↓
Stem Conv (1→64 channels) + BatchNorm + ReLU
        ↓
ResBlock 1: Conv → Conv + Skip Connection (64 channels)
        ↓
Down-sampling Conv (64 → 128 channels)
        ↓
ResBlock 2: Conv → Conv + Skip Connection (128 channels)
        ↓
Adaptive Max Pool (collapses 128 channels to 1 value each)
        ↓
Linear (128→64) → ReLU → Linear (64→1) → Sigmoid
        ↓
Output: probability between 0 and 1
```

---

### Model 2 — BiLSTM + Self-Attention (Deep Learning)

> **Simple Definition:** A model that reads the features like a sentence, both forwards and backwards, then learns which features to focus on.

**LSTM = Long Short-Term Memory**
- A type of neural network with a **"memory cell"** — it can remember information from earlier in the sequence while processing later parts
- Originally designed for text and speech sequences

**Bidirectional (Bi) = reads in both directions**
- One LSTM reads features left → right
- Another reads features right → left
- Both are combined — every feature is understood in the context of all other features

**Self-Attention**
- After reading, the model assigns a **"focus score"** to every position in the sequence
- Positions with high scores contribute more to the final decision
- Think of it as the model **highlighting the most important features** with a yellow marker

**Architecture in this project:**
```
69 PCA components (treated as a sequence of 69 values)
        ↓
BiLSTM Layer 1 (hidden=64, bidirectional → 128 output per step)
        ↓
BiLSTM Layer 2 (hidden=64, bidirectional → 128 output per step)
        ↓
Attention Layer: score = softmax(Linear(128→1)) for each position
        ↓
Weighted Sum: context = Σ (score_i × output_i)
        ↓
Linear (128→64) → ReLU → Linear (64→1) → Sigmoid
        ↓
Output: probability between 0 and 1
```

---

### Model 3 — Random Forest (Traditional ML)

> **Simple Definition:** Builds hundreds of decision trees on random subsets of data and features, then lets them all vote.

**Decision Tree:** A flowchart of yes/no questions.
```
Is Jitter > 0.5?
  YES → Is Shimmer > 0.3? → YES: Parkinson's   NO: Healthy
  NO  → Is HNR < 20?     → YES: Parkinson's   NO: Healthy
```

**The Problem with one tree:** It memorizes the training data (overfits) and performs badly on new patients.

**Random Forest Fix:**
1. Creates **500 different decision trees**
2. Each tree is trained on a **random subset** of patients
3. Each tree uses only a **random subset of features** at each split point
4. All 500 trees vote → the majority wins

**Why it's great here:** Extremely robust on small tabular datasets (756 samples). Doesn't need much tuning. Handles non-linear relationships automatically.

---

### Model 4 — HistGradientBoosting (Traditional ML)

> **Simple Definition:** Builds trees one-by-one, where each new tree specifically learns to fix the mistakes of all previous trees.

**Boosting = Sequential Learning**
```
Tree 1: Makes predictions → some are wrong
Tree 2: Specifically trained to correct Tree 1's errors
Tree 3: Corrects errors of Trees 1+2
...
Tree 500: Very accurate combined model
```

**Hist = Histogram-based**
- Groups continuous feature values into buckets (histograms) for much faster computation
- Far faster than standard gradient boosting with no loss of accuracy

**Why it's great here:** Generally achieves higher accuracy than Random Forest because it actively focuses on the hardest-to-classify patients.

---

### Model 5 — Calibrated SVM (Traditional ML)

> **Simple Definition:** Finds a mathematical boundary that best separates Healthy from Parkinson's patients in a high-dimensional space.

**SVM = Support Vector Machine**
- Finds the **hyperplane** (line in 2D, plane in 3D, hyper-plane in 69D) that separates two classes with the **maximum margin** — the widest gap between the boundary and the nearest data points of each class
- The data points nearest to this boundary are called **"Support Vectors"**

**RBF Kernel = Radial Basis Function**
- The 69 PCA features often can't be separated by a straight line
- The RBF kernel mathematically transforms the data into an even higher-dimensional space where a separating hyperplane CAN be found
- This is the **"kernel trick"** — it works without actually computing the huge transformation

**Calibrated = converts raw scores to true probabilities**
- Standard SVM gives a raw distance from the hyperplane, not a probability (0–1)
- `CalibratedClassifierCV` wraps SVM and converts its scores to proper probabilities using cross-validation

---

## 6. The Ensemble (Combining All 5 Models)

> **Simple Definition:** The 5 models each give their "opinion" (a probability 0–1). Their opinions are averaged together, but smarter models get more say.

### Why combine models?
- CNN1D is excellent at local feature patterns
- BiLSTM catches long-range sequential patterns
- Random Forest & HistGBM are robust on small tabular data
- SVM finds optimal geometric boundaries

No single model is perfect at everything. Combining them covers each other's weaknesses.

---

### AUC-Weighted Soft Voting — Step by Step

**Step 1 — Each model gives a probability (Soft Voting):**
```
CNN1D    says: 0.82  (82% chance of Parkinson's)
BiLSTM   says: 0.78
RF       says: 0.91
HistGBM  says: 0.88
SVM      says: 0.71
```

**Step 2 — Compute each model's AUC on the validation fold:**

> **AUC (Area Under Curve):** A number from 0.5 to 1.0.
> - 1.0 = perfect model
> - 0.5 = random guessing
> Measures how well a model **ranks** sick vs. healthy people, regardless of threshold.

```
CNN1D   AUC: 0.94  →  weight = 0.94 - 0.50 = 0.44
BiLSTM  AUC: 0.91  →  weight = 0.91 - 0.50 = 0.41
RF      AUC: 0.97  →  weight = 0.97 - 0.50 = 0.47
HistGBM AUC: 0.96  →  weight = 0.96 - 0.50 = 0.46
SVM     AUC: 0.89  →  weight = 0.89 - 0.50 = 0.39
```
*(We subtract 0.5 so that a near-random model gets a near-zero weight)*

**Step 3 — Normalize weights to sum to 1:**
```
Total  = 0.44 + 0.41 + 0.47 + 0.46 + 0.39 = 2.17
Weights = [0.203, 0.189, 0.217, 0.212, 0.180]
```

**Step 4 — Weighted average of probabilities:**
```
Final Probability = (0.82×0.203) + (0.78×0.189) + (0.91×0.217) + (0.88×0.212) + (0.71×0.180)
                 = 0.166 + 0.147 + 0.197 + 0.187 + 0.128
                 = 0.825  →  82.5% risk of Parkinson's
```

The Random Forest had the highest AUC (0.97), so its vote carries the most weight. A near-random model (0.51 AUC) would get nearly zero weight and be essentially ignored.

---

## 7. Youden's J Threshold

> **Simple Definition:** Instead of defaulting to "if probability > 50% → Parkinson's", we find the mathematically best cutoff point.

**The Problem with 0.5:**
For imbalanced medical data, 0.5 is usually the wrong cutoff. In medicine, we often care more about catching every sick person (high recall), even if a few healthy people are flagged.

**Youden's J Statistic:**
```
J(threshold) = TPR(threshold) − FPR(threshold)

TPR = True Positive Rate  = % of Parkinson's patients correctly caught
FPR = False Positive Rate = % of Healthy patients incorrectly flagged

Optimal threshold = the value that maximizes J
```

**In our project:** The optimal threshold was found to be **0.5573** (not 0.5). This specific value is saved into `models/ensemble_models.pt` and used consistently every time the app runs.

---

## 8. Cross-Validation (How We Measure Performance Fairly)

> **Simple Definition:** Instead of testing on only 20% of the data, we test on 100% of it — in 5 rounds.

**The Problem:** With only 756 patients, a single random 80/20 split is unreliable. You might get lucky or unlucky with which patients end up in the test set.

**5-Fold Stratified Cross-Validation:**

```
Fold 1: [TEST ✓]  [train]   [train]   [train]   [train]
Fold 2: [train]   [TEST ✓]  [train]   [train]   [train]
Fold 3: [train]   [train]   [TEST ✓]  [train]   [train]
Fold 4: [train]   [train]   [train]   [TEST ✓]  [train]
Fold 5: [train]   [train]   [train]   [train]   [TEST ✓]
```

- **Stratified** = every fold has the same 75% Parkinson's / 25% Healthy ratio
- **Every patient is tested exactly once**
- Average metrics across all 5 folds → reliable, unbiased performance estimate

### Our Final Results:

| Metric | Mean | Std |
|---|---|---|
| **Accuracy** | **91.80%** | ±2.53% |
| **F1-Score** | **92.01%** | ±2.38% |
| **ROC-AUC** | **96.38%** | ±2.28% |

---

## 9. Loss Functions (How Models Learn)

> **Simple Definition:** A loss function measures "how wrong" the model currently is. Training = minimizing this wrongness.

### Focal Loss (used for both neural networks)

**The Problem with standard Cross-Entropy Loss:**
If 75% of samples are Parkinson's, the model learns to predict Parkinson's for everything — this minimizes loss easily, but the model hasn't actually learned anything about the Healthy class.

**Focal Loss Fix:**
```
Focal Loss = −α × (1 − p)^γ × log(p)
```

| Parameter | Meaning |
|---|---|
| `(1 − p)^γ` | Reduces loss contribution for samples the model is already confident about |
| `α = 0.75` | Extra weight given to the positive (Parkinson's) class |
| `γ = 2.0` | Controls how strongly to down-weight easy samples |

**Think of it like:** A teacher who gives simpler homework to students who already understand the material, and focuses harder problems on the students who are still struggling.

### Training Settings for Neural Networks

| Setting | Value | Why |
|---|---|---|
| Optimizer | AdamW | Adaptive learning rate + weight decay to prevent overfitting |
| Max Epochs | 200 | Maximum training rounds per fold |
| Early Stopping | Patience = 25 | Stops if validation loss doesn't improve for 25 epochs |
| LR Scheduler | CosineAnnealingLR | Smoothly reduces learning rate over training |
| Gradient Clipping | 1.0 | Prevents exploding gradients in deep networks |

---

## 10. What Gets Saved (The Artifacts)

After training, these files are saved in `models/`:

| File | Contains | Used For |
|---|---|---|
| `scaler.pkl` | StandardScaler fitted on 753 features | Normalize new patient data the same way |
| `boruta_mask.pkl` | 753 True/False values — which features to keep | Filter new data to the same 131 features |
| `pca.pkl` | PCA transformer (131 → 69 dimensions) | Compress new data the same way |
| `ensemble_models.pt` | CNN1D + BiLSTM weights + AUC weights + optimal threshold | Deep learning inference |
| `ml_ensemble.pkl` | Trained RF + HistGBM + SVM | Traditional ML inference |
| `cv_results.json` | Accuracy / F1 / AUC per fold | Display in the web app sidebar |

---

## 11. The Web App (app.py)

Built with **Streamlit** — a Python library that turns Python scripts into interactive web apps.

### Inference Flow (what happens when you upload a CSV):

```
1. Upload CSV (753 feature columns per patient row)
        ↓
2. Load scaler.pkl       → normalize your data
3. Apply boruta_mask.pkl → keep only the 131 important features
4. Apply pca.pkl         → compress to 69 dimensions
        ↓
5. Load CNN1D  from ensemble_models.pt → probability for each patient
   Load BiLSTM from ensemble_models.pt → probability for each patient
   Load RF     from ml_ensemble.pkl    → probability for each patient
   Load HistGBM from ml_ensemble.pkl   → probability for each patient
   Load SVM    from ml_ensemble.pkl    → probability for each patient
        ↓
6. AUC-weighted average of 5 probabilities
        ↓
7. Apply saved Youden's J threshold (0.5573)
        ↓
8. Display result:
   - Parkinson's / Healthy banner
   - Risk % | Healthy % | Confidence %
   - Per-model probability breakdown with weights
   - Download predictions as CSV
```

**Key Design Principle:** Steps 2, 3, 4 in inference use the **exact same saved objects** that were created during training. This guarantees 100% consistency — the new patient's data is processed in exactly the same way as the training data was.

---

## 12. The Complete Picture (Everything Together)

```
╔══════════════════════════════════════════════════════════════════════╗
║           TRAINING TIME  (python src/advanced_model.py)             ║
╚══════════════════════════════════════════════════════════════════════╝

756 patients × 753 voice features
        │
        ▼
[StandardScaler] Zero-mean, unit-variance → same scale for all features
        │
        ▼
[Boruta] Statistical test → keep 131 of 753 features
        │
        ▼
[PCA]   Compress 131 → 69 components (99% information retained)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│         5-FOLD STRATIFIED CV LOOP (×5)                  │
│                                                         │
│  [ADASYN] Balance training set: ~604 → ~876 samples     │
│                                                         │
│  ├── Train Residual CNN1D     (Focal Loss, 200 epochs)  │
│  ├── Train BiLSTM-Attention   (Focal Loss, 200 epochs)  │
│  ├── Train Random Forest      (500 trees)               │
│  ├── Train HistGradientBoost  (500 iterations)          │
│  └── Train Calibrated SVM     (RBF kernel)              │
│                                                         │
│  Compute each model's AUC on validation fold            │
│  → AUC-weighted soft voting                             │
│  → Find Youden's J optimal threshold                    │
│  → Record metrics (Acc / Precision / Recall / F1 / AUC) │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Average metrics across 5 folds
→ Accuracy: 91.8%  F1: 92.0%  AUC: 96.4%

Save best fold's models:
→ scaler.pkl  boruta_mask.pkl  pca.pkl
→ ensemble_models.pt  (CNN1D + BiLSTM weights + threshold)
→ ml_ensemble.pkl     (RF + HistGBM + SVM)
→ cv_results.json


╔══════════════════════════════════════════════════════════════════════╗
║       INFERENCE TIME  (streamlit run app.py)                        ║
╚══════════════════════════════════════════════════════════════════════╝

New patient CSV uploaded
        │
        ▼
Load scaler.pkl    → normalize
Load boruta_mask   → filter to 131 features
Load pca.pkl       → compress to 69 dimensions
        │
        ▼
Load ensemble_models.pt → CNN1D prob + BiLSTM prob
Load ml_ensemble.pkl    → RF prob + HistGBM prob + SVM prob
        │
        ▼
Weighted average (using saved AUC weights)
        │
        ▼
Compare to saved threshold (0.5573)
        │
        ▼
Display: ✅ Healthy  OR  ⚠️ Parkinson's
+ Risk % + per-model confidence breakdown
```

---

## 13. Quick Glossary (All Terms in One Place)

| Term | One-Line Definition |
|---|---|
| **Accuracy** | % of predictions that are correct overall |
| **ADASYN** | Creates artificial minority-class samples to balance training data |
| **AUC (ROC-AUC)** | How well a model ranks sick vs. healthy (1.0 = perfect, 0.5 = random guessing) |
| **Attention** | Mechanism that lets a model focus on the most relevant parts of input |
| **BiLSTM** | LSTM that reads data both forwards and backwards |
| **Boruta** | Feature selection algorithm that identifies truly important features by comparing them to random shuffled copies |
| **CNN** | Neural network that scans input with sliding filters to detect local patterns |
| **Calibrated SVM** | SVM whose raw scores are converted to proper probabilities |
| **Cross-Validation** | Testing every sample exactly once by rotating which fold is the test set |
| **Decision Threshold** | The cutoff probability above which we predict "Parkinson's" |
| **F1-Score** | Harmonic mean of Precision and Recall — good metric for imbalanced data |
| **False Positive** | Predicted Parkinson's, but the person is actually healthy |
| **Focal Loss** | Loss function that focuses learning on hard, misclassified examples |
| **Gradient Boosting** | Builds trees sequentially, each fixing the previous one's mistakes |
| **HistGradientBoosting** | Fast histogram-based version of gradient boosting |
| **HNR** | Harmonic-to-Noise Ratio — measure of voice clarity |
| **LSTM** | Neural network with memory cells that handle sequential data |
| **MFCC** | Mel-Frequency Cepstral Coefficients — compact audio fingerprint |
| **Oversampling** | Creating extra samples of the minority class to fix class imbalance |
| **PCA** | Compresses features into fewer, uncorrelated dimensions while retaining variance |
| **Precision** | Of all Parkinson's predictions, what fraction were actually Parkinson's? |
| **Random Forest** | Hundreds of decision trees voting together |
| **Recall / Sensitivity** | Of all actual Parkinson's patients, what fraction did we catch? |
| **Residual Connection** | Shortcut that adds input directly to output of a layer: `out = f(x) + x` |
| **ROC Curve** | Graph of True Positive Rate vs False Positive Rate at all thresholds |
| **SVM** | Finds the widest possible mathematical boundary between two classes |
| **Soft Voting** | Average the probability scores (not just yes/no votes) from multiple models |
| **StandardScaler** | Transforms features to have mean = 0, standard deviation = 1 |
| **Stratified** | Maintaining class ratios when splitting data into folds |
| **Support Vectors** | The data points closest to the SVM decision boundary |
| **Youden's J** | Formula `TPR − FPR` used to find the optimal decision threshold |

---

*This project achieves **91.8% Accuracy** and **96.4% ROC-AUC** on the UCI Parkinson's Speech Dataset via 5-Fold Stratified Cross-Validation using a hybrid deep learning + traditional ML ensemble.*
