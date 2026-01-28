# train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib  # Để save model

train_df = pd.read_csv('../data/train_scaled.csv')
X_train = train_df.drop('diagnosis', axis=1)
y_train = train_df['diagnosis']

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')  # Save model
print("Model trained and saved!")