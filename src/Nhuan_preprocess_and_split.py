"""
Preprocessing Pipeline for Logistic Regression
- Feature Selection
- Train / Test Split (80 / 20)
- Feature Scaling

Dataset: WDBC (Breast Cancer)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load raw data
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# 2. Encode target variable
df['y'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 3. Feature Selection
selected_features = [
    'radius_mean',
    'texture_mean',
    'smoothness_mean',
    'concavity_mean',
    'symmetry_mean',
    'radius_worst',
    'area_worst'
]

X = df[selected_features]
y = df['y']


# 4. Train / Test Split
# 80% Train - 20% Test
# Stratified to preserve class distribution
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------------------------------
# 5. Feature Scaling (StandardScaler)
# Fit ONLY on training data
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Save processed datasets
# -------------------------------
train_df = pd.DataFrame(X_train_scaled, columns=selected_features)
train_df['y'] = y_train.values

test_df = pd.DataFrame(X_test_scaled, columns=selected_features)
test_df['y'] = y_test.values

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

print("Saved train and test datasets to data/")
print("✅ Preprocessing completed.")
