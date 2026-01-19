import os
import numpy as np
import pandas as pd


# ===============================
# Logistic Regression Functions
# ===============================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, weights, bias, threshold=0.5):
    z = np.dot(X, weights) + bias
    probs = sigmoid(z)
    return (probs >= threshold).astype(int)


# ===============================
# Metrics
# ===============================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def precision(y_true, y_pred):
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp + 1e-9)


def recall(y_true, y_pred):
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn + 1e-9)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-9)


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
# Evaluate
# ===============================
y_pred = predict(X_test, weights, bias)

tp, tn, fp, fn = confusion_matrix(y_test, y_pred)

print("Accuracy :", accuracy(y_test, y_pred))
print("Precision:", precision(y_test, y_pred))
print("Recall   :", recall(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix")
print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)
