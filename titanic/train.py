import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
df = pd.read_csv('titanic.csv')

# 2. Preprocessing & Feature Engineering
# Remove high-cardinality/unnecessary columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle Missing Values safely
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert Categorical features to numbers (dtype=int ensures 1/0 instead of True/False)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True, dtype=int)

# Separate Target (y) and Features (X)
X = df.drop('Survived', axis=1)
y = df['Survived']

# 3. Train-Test Split (CORRECTED: Using test_size instead of test_test_split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 6. Print Verification Evaluation
y_pred = model.predict(X_test_scaled)
print(f"Model trained successfully! Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. Export operational artifacts
joblib.dump(model, 'titanic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print("All artifacts successfully saved to disk!")