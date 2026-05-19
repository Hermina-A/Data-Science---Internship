import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error, r2_score, 
    confusion_matrix, classification_report
)

def load_data(filepath="employee_salary_dataset.csv"):
    """Loads the dataset and handles missing values/duplicates."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{filepath}' not found. Please run Step 1 to generate it.")
        
    df = pd.read_csv(filepath)
    
    # Clean duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle missing values structurally if any exist
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    return df

def build_preprocessing_pipeline(X):
    """Creates a ColumnTransformer to scale numeric inputs and encode categorical text."""
    num_features = ['Age', 'Experience', 'Working_Hours', 'Performance_Score', 'Projects_Completed']
    cat_features = ['Gender', 'Education', 'Department']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
        ]
    )
    return preprocessor, num_features, cat_features

def run_evaluation_suite(y_test_reg, y_pred_lr, y_pred_rf_reg, y_test_clf, y_pred_log, y_pred_rf_clf):
    """Calculates, formats, and displays model evaluation metrics."""
    print("\n" + "="*50)
    print("         REGRESSION METRICS (EXACT SALARY)        ")
    print("="*50)
    
    reg_metrics = {
        'Metric': ['MAE', 'RMSE', 'R² Score'],
        'Linear Regression': [
            f"${mean_absolute_error(y_test_reg, y_pred_lr):,.2f}",
            f"${np.sqrt(mean_squared_error(y_test_reg, y_pred_lr)):,.2f}",
            f"{r2_score(y_test_reg, y_pred_lr):.4f}"
        ],
        'Random Forest Regressor': [
            f"${mean_absolute_error(y_test_reg, y_pred_rf_reg):,.2f}",
            f"${np.sqrt(mean_squared_error(y_test_reg, y_pred_rf_reg)):,.2f}",
            f"{r2_score(y_test_reg, y_pred_rf_reg):.4f}"
        ]
    }
    print(pd.DataFrame(reg_metrics).to_string(index=False))
    
    print("\n" + "="*50)
    print("       CLASSIFICATION METRICS (SALARY BRACKET)     ")
    print("="*50)
    
    clf_metrics = {
        'Metric': ['Accuracy Score'],
        'Logistic Regression': [f"{accuracy_score(y_test_clf, y_pred_log) * 100:.2f}%"],
        'Random Forest Classifier': [f"{accuracy_score(y_test_clf, y_pred_rf_clf) * 100:.2f}%"]
    }
    print(pd.DataFrame(clf_metrics).to_string(index=False))

def main():
    print("🔄 Initializing Model Training Pipeline...")
    
    # 1. Load Data
    df = load_data("employee_salary_dataset.csv")
    
    # 2. Split Features and Target Vector
    X = df.drop(columns=['Employee_ID', 'Salary'])
    y = df['Salary']
    
    # 3. Create Categorical Targets for Classification Tasks (Using Quantiles)
    low_thresh = y.quantile(0.33)
    high_thresh = y.quantile(0.66)
    
    def categorize_salary(val):
        if val <= low_thresh: return 'Low Salary'
        elif val <= high_thresh: return 'Medium Salary'
        return 'High Salary'
        
    y_cat = y.apply(categorize_salary)
    
    # 4. Fit Preprocessing Pipeline Transformation Artifacts
    preprocessor, num_features, cat_features = build_preprocessing_pipeline(X)
    X_processed = preprocessor.fit_transform(X)
    
    # Get structured column tracking positions
    encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    all_feature_names = num_features + list(encoded_cat_names)
    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
    
    # 5. Train-Test Splits
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X_processed_df, y, test_size=0.2, random_state=42)
    _, _, y_train_clf, y_test_clf = train_test_split(X_processed_df, y_cat, test_size=0.2, random_state=42)
    
    # 6. Train Models
    print("🏋️‍♂️ Training Machine Learning models...")
    
    # A. Linear Regression
    lr_model = LinearRegression().fit(X_train, y_train_reg)
    y_pred_lr = lr_model.predict(X_test)
    
    # B. Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_reg)
    y_pred_rf_reg = rf_regressor.predict(X_test)
    
    # C. Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train_clf)
    y_pred_log = log_reg.predict(X_test)
    
    # D. Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train_clf)
    y_pred_rf_clf = rf_classifier.predict(X_test)
    
    # 7. Evaluate Performance Outputs
    run_evaluation_suite(y_test_reg, y_pred_lr, y_pred_rf_reg, y_test_clf, y_pred_log, y_pred_rf_clf)
    
    # 8. Export Production-Ready Model Artifacts
    print("\n📦 Exporting binary model files to local directory...")
    artifacts = {
        'preprocessor': preprocessor,
        'linear_regression': lr_model,
        'random_forest_regressor': rf_regressor,
        'logistic_regression': log_reg,
        'random_forest_classifier': rf_classifier,
        'classification_thresholds': {'low': low_thresh, 'high': high_thresh}
    }
    
    with open("salary_model_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
        
    print("✅ System training successful! File saved as: 'salary_model_artifacts.pkl'")

if __name__ == "__main__":
    main()

