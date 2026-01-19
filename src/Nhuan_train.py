import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Logistic Regression - Scratch
# ===============================

def sigmoid(z):
    """
    Sigmoid function with numerical stability
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_loss(y_true, y_pred):
    """
    Binary Cross-Entropy Loss
    """
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )
    return loss


def predict_proba(X, weights, bias):
    """
    Predict probability P(y=1|x)
    """
    return sigmoid(np.dot(X, weights) + bias)


def train_logistic_regression(X_train, y_train, X_val, y_val,
                              lr=0.01, n_epochs=4000):
    """
    Train Logistic Regression using Gradient Descent
    """
    n_samples, n_features = X_train.shape

    # Initialize parameters
    weights = np.zeros(n_features)
    bias = 0.0

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # -----------------
        # Forward pass
        # -----------------
        z = np.dot(X_train, weights) + bias
        y_pred = sigmoid(z)

        # -----------------
        # Compute loss
        # -----------------
        train_loss = compute_loss(y_train, y_pred)
        train_losses.append(train_loss)

        # Validation loss
        y_val_pred = predict_proba(X_val, weights, bias)
        val_loss = compute_loss(y_val, y_val_pred)
        val_losses.append(val_loss)

        # -----------------
        # Backward pass
        # -----------------
        dw = (1 / n_samples) * np.dot(X_train.T, (y_pred - y_train))
        db = (1 / n_samples) * np.sum(y_pred - y_train)

        # -----------------
        # Update parameters
        # -----------------
        weights -= lr * dw
        bias -= lr * db

        # -----------------
        # Logging
        # -----------------
        if epoch % 200 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

    return weights, bias, train_losses, val_losses


# ===============================
# Load data
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))

X_train = train_df.drop("y", axis=1).values
y_train = train_df["y"].values

X_val = val_df.drop("y", axis=1).values
y_val = val_df["y"].values

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# ===============================
# Train model
# ===============================
weights, bias, train_losses, val_losses = train_logistic_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    lr=0.01,
    n_epochs=4000
)

# ===============================
# Save model parameters
# ===============================
np.save(os.path.join(DATA_DIR, "weights.npy"), weights)
np.save(os.path.join(DATA_DIR, "bias.npy"), bias)

print("Model parameters saved successfully.")

# ===============================
# Plot loss curves
# ===============================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training & Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()
