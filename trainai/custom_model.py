import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
        # --- THÊM: Từ điển lưu lịch sử để vẽ biểu đồ ---
        self.history = {
            'loss': [],
            'w': [],         # Lưu trọng số
            'b': [],         # Lưu bias
            'z_sample': [],  # Lưu giá trị z của mẫu đầu tiên
            'y_hat_sample': [] # Lưu xác suất dự đoán của mẫu đầu tiên
        }

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # --- THÊM: GHI LOG LỊCH SỬ ---
            # 1. Tính Loss
            loss = -np.mean(y * np.log(y_predicted + 1e-15) + (1-y) * np.log(1-y_predicted + 1e-15))
            self.history['loss'].append(loss)
            
            # 2. Lưu tham số (dùng .copy() để không bị tham chiếu)
            self.history['w'].append(self.weights.copy())
            self.history['b'].append(self.bias)
            
            # 3. Theo dõi mẫu dữ liệu đầu tiên (Sample 0)
            self.history['z_sample'].append(linear_model[0])
            self.history['y_hat_sample'].append(y_predicted[0])
            # -----------------------------

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                 print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])

    def predict_proba(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)