import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# =========================
# 1. Load dataset
# =========================
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "data.csv")

df = pd.read_csv(INPUT_FILE)

# =========================
# 2. Drop ID column
# =========================
df = df.drop(columns=["id"])

# =========================
# 3. Tách feature và label
# =========================
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].map({"M": 1, "B": 0})

# =========================
# 4. Chia train / test (stratified)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5. Chuẩn hóa dữ liệu
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 6. Gộp lại để lưu file
# =========================
train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
train_df["diagnosis"] = y_train.values

test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df["diagnosis"] = y_test.values

# =========================
# 7. Lưu file ra thư mục data
# =========================
TRAIN_FILE = os.path.join(DATA_DIR, "train_scaled.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_scaled.csv")

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# =========================
# 8. Kiểm tra kết quả
# =========================
print("✅ Đã xử lý và lưu dữ liệu thành công")
print(f"- Train file: {TRAIN_FILE}")
print(f"- Test file : {TEST_FILE}")

print("\nTrain size:", train_df.shape[0])
print("Test size :", test_df.shape[0])

print("\nClass distribution (train):")
print(train_df["diagnosis"].value_counts(normalize=True))

print("\nClass distribution (test):")
print(test_df["diagnosis"].value_counts(normalize=True))
