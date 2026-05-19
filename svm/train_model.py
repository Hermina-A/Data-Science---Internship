import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Loading Iris dataset...")
# 1. Load the dataset
df = pd.read_csv('Iris.csv')

# 2. Separate features (X) and target/label (y)
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# 3. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest Classifier
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Check accuracy
accuracy = model.score(X_test, y_test)
print(f"Model trained successfully! Test Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model to a file
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'iris_model.pkl'")