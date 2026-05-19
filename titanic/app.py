import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page layout setup
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# Load model assets safely with caching
@st.cache_resource
def load_ml_assets():
    model = joblib.load('titanic_model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, scaler, columns

try:
    model, scaler, model_columns = load_ml_assets()
except FileNotFoundError:
    st.error("⚠️ Operational files missing! Run 'python train.py' in your terminal first.")
    st.stop()

# --- User Interface Layout ---
st.title("🚢 Titanic Passenger Survival Predictor")
st.markdown("Enter details below to compute passenger survival probabilities using a trained Logistic Regression pipeline.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket Class", [1, 2, 3], format_func=lambda x: f"Class {x}")
    age = st.slider("Passenger Age", min_value=1, max_value=100, value=28)
    fare = st.slider("Fare Paid (£)", min_value=0.0, max_value=512.0, value=32.2)

with col2:
    sex = st.selectbox("Gender", ["Female", "Male"])
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)

embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])

st.markdown("---")

# --- Inference Engine ---
if st.button("Calculate Survival Probability", type="primary"):
    # Generate an empty single-row DataFrame with the exact trained schema
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Map input selections to structured columns
    input_df['Pclass'] = int(pclass)
    input_df['Age'] = float(age)
    input_df['SibSp'] = int(sibsp)
    input_df['Parch'] = int(parch)
    input_df['Fare'] = float(fare)
    
    # Handle explicit binary configurations 
    if sex == "Male":
        input_df['Sex_male'] = 1
        
    if embarked == "Queenstown":
        input_df['Embarked_Q'] = 1
    elif embarked == "Southampton":
        input_df['Embarked_S'] = 1
        
    # Standardize data and perform inferencing
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    survival_chance = model.predict_proba(scaled_input)[0][1] * 100

    # Render metrics to frontend dashboard
    if prediction == 1:
        st.success(f"🎉 **Result: Predicted to Survive!**")
        st.metric(label="Calculated Survival Probability", value=f"{survival_chance:.2f}%")
    else:
        st.error(f"💔 **Result: Predicted Not to Survive.**")
        st.metric(label="Calculated Survival Probability", value=f"{survival_chance:.2f}%")
