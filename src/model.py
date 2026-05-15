import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# --- 1. Data Loading and Preparation (Needed to define X_train, y_train) ---

# Fetch dataset from UCI ML Repository (ID 174)
parkinsons = fetch_ucirepo(id=174)

# Separate features (X) and label (y)
X = parkinsons.data.features
y = parkinsons.data.targets

# Ensure y is a Series/1D array
if isinstance(y, pd.DataFrame):
    target_column = y.columns[0]
    y = y[target_column].squeeze()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance data with SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Train-test split
# We split the balanced data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2, # We only need X_train, y_train for training, but the split is good practice
    random_state=42,
    stratify=y_balanced
)

# --- 2. Model Training and Hyperparameter Tuning ---

# Define the hyperparameter grid for the Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}
rf = RandomForestClassifier(random_state=42)

# Use GridSearchCV for hyperparameter tuning
print("Starting Grid Search...")
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# --- 3. Output and Model Saving ---

print("\n--- Training Results ---")
print("Best hyperparameters found:", grid.best_params_)
print("Cross-Validation Accuracy (Best Score): {:.4f}".format(grid.best_score_))

# Save the best model estimator (the one with the best params)
model_filename = 'parkinsons_rf_model.pkl'
joblib.dump(grid.best_estimator_, model_filename)

print(f"\nModel successfully saved as '{model_filename}'")