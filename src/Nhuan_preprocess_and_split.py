"""
Preprocessing Pipeline for Logistic Regression
- Feature Selection
- Train / Validation / Test Split
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

# -------------------------------
# 2. Encode target variable
# M = 1 (Malignant), B = 0 (Benign)
# -------------------------------
df['y'] = df['diagnosis'].map({'M': 1, 'B': 0})

# -------------------------------
# 3. Feature Selection (based on EDA)
# -------------------------------
selected_features = [
    'radius_mean',        # Ban kinh trung binh cua khoi u, phan biet ro giua M va B
    'texture_mean',       # Do bien thien ket cau, giup mo hinh phan tach hai lop
    'smoothness_mean',    # Do nhan be mat, lien quan truc tiep toi tinh ac tinh
    'concavity_mean',     # Muc do lom vao cua khoi u, dac trung quan trong cua ung thu
    'symmetry_mean',      # Do doi xung hinh dang, khoi u ac tinh thuong kem doi xung
    'radius_worst',       # Gia tri lon nhat cua ban kinh, the hien kich thuoc cuc doan
    'area_worst'          # Dien tich lon nhat, tuong quan manh voi muc do ac tinh
]

X = df[selected_features]
y = df['y']

# -------------------------------
# 4. Train / Validation / Test Split
# 70% / 15% / 15%
# Stratified to preserve class distribution
# -------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# -------------------------------
# 5. Feature Scaling (StandardScaler)
# Fit ONLY on training data
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Save processed datasets
# -------------------------------
train_df = pd.DataFrame(X_train_scaled, columns=selected_features)
train_df['y'] = y_train.values

val_df = pd.DataFrame(X_val_scaled, columns=selected_features)
val_df['y'] = y_val.values

test_df = pd.DataFrame(X_test_scaled, columns=selected_features)
test_df['y'] = y_test.values

DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

print("Saved train / val / test datasets to data/")

print("✅ Preprocessing completed. Processed files saved to data/processed/")
