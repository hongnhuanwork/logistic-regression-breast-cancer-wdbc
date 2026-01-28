# evaluate.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

test_df = pd.read_csv('../data/test_scaled.csv')
X_test = test_df.drop('diagnosis', axis=1)
y_test = test_df['diagnosis']

model = joblib.load('model.pkl')
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics (in ra console)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# ... (thêm precision, recall, etc. như notebook)

# Plot & save
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save hình
plt.show()