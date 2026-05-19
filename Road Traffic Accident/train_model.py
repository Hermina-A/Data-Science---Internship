import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

def load_data(filepath):
    """Loads and runs basic structural cleaning on the dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source dataset not found at: {filepath}")
        
    print(f"Loading dataset from: {filepath}...")
    df = pd.read_csv(filepath)
    
    # Standardize string missing indicators to actual NaNs
    df.replace("na", np.nan, inplace=True)
    
    # Engineering hour of day from standard time format
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
    df["Hour_of_day"] = df["Time"].dt.hour
    df["Hour_of_day"] = df["Hour_of_day"].fillna(df["Hour_of_day"].median())
    df.drop(columns=["Time"], inplace=True, errors="ignore")
    
    return df

def train_pipeline():
    # 1. Load configuration variables
    data_path = "RTA Dataset.csv"
    model_output_path = "rta_rf_model.pkl"
    encoder_output_path = "rta_encoder.pkl"
    
    features = [
        "Day_of_week",
        "Age_band_of_driver",
        "Driving_experience",
        "Light_conditions",
        "Weather_conditions",
        "Hour_of_day",
        "Cause_of_accident"
    ]
    target = "Accident_severity"
    
    # 2. Extract dataframe and isolate matrices
    df = load_data(data_path)
    X = df[features].copy()
    y = df[target].copy()
    
    # 3. Preprocess missing entries in categorical features
    cat_cols = X.select_dtypes(include=["object"]).columns
    X[cat_cols] = X[cat_cols].fillna("Unknown")
    
    # 4. Pipeline feature transformation via Ordinal Encoder
    print("Fitting categorical data encoders...")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_encoded = X.copy()
    X_encoded[X.columns] = encoder.fit_transform(X)
    
    # 5. Define and train the structural ensemble architecture
    print("Training Random Forest Classifier (Optimizing Class Weights)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced", 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_encoded, y)
    
    # 6. Serialize and save tracking artifacts
    print(f"Saving trained model to: {model_output_path}")
    joblib.dump(model, model_output_path)
    
    print(f"Saving encoder configuration to: {encoder_output_path}")
    joblib.dump(encoder, encoder_output_path)
    
    print("\nModel pipeline training sequence finished successfully.")

if __name__ == "__main__":
    train_pipeline()