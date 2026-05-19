import streamlit as st
import pandas as pd
import pickle

# Set page title and layout
st.set_page_config(page_title="Iris Species Predictor", layout="centered")

# Load the trained model using caching so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    with open('iris_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'iris_model.pkl' not found. Please run 'train_model.py' first!")
    st.stop()

# Title and description
st.title("🌸 Iris Flower Species Predictor")
st.write("""
This app predicts the **Iris flower species** based on user-provided input measurements using a Machine Learning model.
""")

# Sidebar interface for feature inputs
st.sidebar.header("Input Flower Measurements")

def user_input_features():
    # Min/Max ranges chosen based on actual distribution of the Iris dataset
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.3, 7.9, 5.8)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 6.9, 3.8)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.2)
    
    data = {
        'SepalLengthCm': sepal_length,
        'SepalWidthCm': sepal_width,
        'PetalLengthCm': petal_length,
        'PetalWidthCm': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Store user inputs into a dataframe
input_df = user_input_features()

# Display user input values on main panel
st.subheader('Selected Input Values')
st.write(input_df)

# Prediction Section
st.markdown("---")
if st.button('Predict Species', type='primary'):
    # Get class name prediction
    prediction = model.predict(input_df)[0]
    
    # Get probability mapping
    prediction_proba = model.predict_proba(input_df)[0]
    
    # Show predicted result
    st.subheader('Prediction Result')
    st.success(f"The model predicts this flower belongs to the species: **{prediction}**")
    
    # Show confidence probabilities
    st.subheader('Prediction Confidence (Probability)')
    prob_df = pd.DataFrame([prediction_proba], columns=model.classes_)
    st.dataframe(prob_df.style.format("{:.2%}"))
