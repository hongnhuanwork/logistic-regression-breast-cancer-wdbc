import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Khởi tạo mô hình.
        learning_rate: Tốc độ học (alpha)
        n_iterations: Số lần lặp để tối ưu (epochs)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Hàm kích hoạt Sigmoid: chuyển giá trị thực về khoảng (0, 1)"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Huấn luyện mô hình dùng Gradient Descent"""
        # Chuyển dữ liệu về dạng numpy array để tính toán ma trận
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape

        # 1. Khởi tạo tham số (Trọng số w và hệ số tự do b) bằng 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Vòng lặp tối ưu hóa (Gradient Descent)
        for i in range(self.n_iterations):
            # a. Tính toán tuyến tính: z = w*X + b
            linear_model = np.dot(X, self.weights) + self.bias
            
            # b. Áp dụng hàm Sigmoid để ra xác suất dự đoán y_pred
            y_predicted = self._sigmoid(linear_model)

            # c. Tính đạo hàm (Gradient)
            # dw = (1/m) * X.T * (y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # db = (1/m) * sum(y_pred - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # d. Cập nhật tham số: w = w - learning_rate * dw
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # In loss mỗi 100 vòng lặp để theo dõi
            if i % 100 == 0:
                 loss = -np.mean(y * np.log(y_predicted) + (1-y) * np.log(1-y_predicted))
                 print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        """Dự đoán nhãn (0 hoặc 1)"""
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        
        # Nếu xác suất > 0.5 thì là 1 (Ác tính), ngược lại là 0
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def predict_proba(self, X):
        """Dự đoán xác suất (cho việc vẽ ROC curve hoặc ngưỡng khác)"""
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)