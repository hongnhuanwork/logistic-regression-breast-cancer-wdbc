import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Logistic Regression Functions
# ===============================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_proba(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)


def predict(X, weights, bias, threshold=0.5):
    probs = predict_proba(X, weights, bias)
    return (probs >= threshold).astype(int)


# ===============================
# Metrics
# ===============================

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(tp, fp):
    return tp / (tp + fp + 1e-9)


def recall(tp, fn):
    return tp / (tp + fn + 1e-9)


def f1_score(p, r):
    return 2 * p * r / (p + r + 1e-9)


# ===============================
# ROC & AUC (scratch)
# ===============================

def roc_curve(y_true, y_prob):
    thresholds = np.linspace(0, 1, 100)
    tpr_list, fpr_list = [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp, tn, fp, fn = confusion_matrix(y_true, y_pred)

        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list)


def auc_score(fpr, tpr):
    return np.trapz(tpr, fpr)


# ===============================
# Load data
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

X_test = test_df.drop("y", axis=1).values
y_test = test_df["y"].values

# ===============================
# Load model
# ===============================

weights = np.load(os.path.join(DATA_DIR, "weights.npy"))
bias = np.load(os.path.join(DATA_DIR, "bias.npy"))

# ===============================
# Prediction
# ===============================

y_prob = predict_proba(X_test, weights, bias)
y_pred = predict(X_test, weights, bias)

tp, tn, fp, fn = confusion_matrix(y_test, y_pred)

acc = accuracy(y_test, y_pred)
prec = precision(tp, fp)
rec = recall(tp, fn)
f1 = f1_score(prec, rec)

# ===============================
# Print academic-style report
# ===============================

print("===== MODEL EVALUATION REPORT =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nConfusion Matrix")
print(f"TP: {tp} | FP: {fp}")
print(f"FN: {fn} | TN: {tn}")

# ===============================
# Plot Confusion Matrix
# ===============================

plt.figure()
cm = np.array([[tp, fp],
               [fn, tn]])

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xticks([0, 1], ["Predicted M", "Predicted B"])
plt.yticks([0, 1], ["Actual M", "Actual B"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.show()

# ===============================
# ROC Curve
# ===============================

fpr, tpr = roc_curve(y_test, y_prob)
auc = auc_score(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
