import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import glob

# --- 1. Data Loading and Preparation ---

# Use glob to handle the special apostrophe in the directory name
csv_matches = glob.glob('C:/Users/PREETI/Desktop/Parkinson*/data/pd_speech_features.csv')
if not csv_matches:
    # Fallback: resolve relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, 'data', 'pd_speech_features.csv')
else:
    data_path = csv_matches[0]

project_dir = os.path.dirname(os.path.dirname(data_path))

print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path, header=1)  # Use second row as column names

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['class'].value_counts()}")

# Drop the 'id' column as it's just an identifier, not a feature
X = df.drop(columns=['id', 'class'])
y = df['class']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class balance in splits
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# --- 2. Model Training ---

print("\nTraining Random Forest Classifier...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# --- 3. Evaluation ---

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n" + "=" * 50)
print(f"  ACCURACY: {accuracy * 100:.2f}%")
print("=" * 50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Parkinson (1)']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- 4. Save Model and Scaler ---

model_filename = os.path.join(project_dir, 'parkinsons_rf_model.pkl')
scaler_filename = os.path.join(project_dir, 'scaler.pkl')

joblib.dump(rf, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"\nModel saved as '{model_filename}'")
print(f"Scaler saved as '{scaler_filename}'")
