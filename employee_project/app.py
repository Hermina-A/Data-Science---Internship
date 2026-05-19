import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

st.set_page_config(page_title="Employee Salary Prediction Dashboard", layout="wide")
st.title("💼 Employee Salary Prediction & Classification System")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('employee_data.csv')

df = load_data()

# Preprocessing Pipeline for Models
le_gender = LabelEncoder().fit(['Male', 'Female'])
le_edu = LabelEncoder().fit(['Bachelor', 'Master', 'PhD'])
le_dept = LabelEncoder().fit(['IT', 'HR', 'Finance', 'Marketing', 'Sales'])

def preprocess_df(data):
    df_mod = data.copy()
    df_mod['Gender'] = le_gender.transform(df_mod['Gender'])
    df_mod['Education'] = le_edu.transform(df_mod['Education'])
    df_mod['Department'] = le_dept.transform(df_mod['Department'])
    
    def categorize_salary(sal):
        if sal < 60000: return 0  # Low
        elif sal <= 110000: return 1  # Medium
        else: return 2  # High
        
    df_mod['Salary_Category'] = df_mod['Salary'].apply(categorize_salary)
    return df_mod

df_mod = preprocess_df(df)
X = df_mod.drop(['Employee_ID', 'Salary', 'Salary_Category'], axis=1)
y_reg = df_mod['Salary']
y_clf = df_mod['Salary_Category']

X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
lr = LinearRegression().fit(X_train_scaled, y_train_reg)
rf = RandomForestRegressor(random_state=42).fit(X_train_scaled, y_train_reg)
log_reg = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train_clf)

# Sidebar UI for Input Form
st.sidebar.header("User Input Features")
input_age = st.sidebar.slider("Age", 22, 65, 30)
input_gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
input_edu = st.sidebar.selectbox("Education Level", ['Bachelor', 'Master', 'PhD'])
input_dept = st.sidebar.selectbox("Department", ['IT', 'HR', 'Finance', 'Marketing', 'Sales'])
input_exp = st.sidebar.slider("Years of Experience", 0, 40, 5)
input_hours = st.sidebar.slider("Weekly Working Hours", 35, 60, 40)
input_perf = st.sidebar.slider("Performance Score", 1, 4, 3)
input_projects = st.sidebar.slider("Projects Completed", 0, 50, 10)

# Layout Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dataset & Visualizations", "🤖 Real-Time Prediction", "📈 Model Evaluation"])

with tab1:
    st.subheader("Generated Employee Dataset (First 10 Rows)")
    st.dataframe(df.head(10))
    
    st.subheader("Data Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['Salary'], kde=True, color='skyblue', ax=ax)
        ax.set_title("Salary Distribution")
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        sns.scatterplot(x='Experience', y='Salary', data=df, hue='Performance_Score', palette='viridis', ax=ax)
        ax.set_title("Experience vs Salary")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.heatmap(df_mod.drop(['Employee_ID'], axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Department', y='Salary', data=df, ax=ax, errorbar=None)
        ax.set_title("Average Salary by Department")
        st.pyplot(fig)

with tab2:
    st.subheader("Predict Single Employee Profile")
    
    # Process inputs
    user_data = pd.DataFrame([{
        'Age': input_age,
        'Gender': le_gender.transform([input_gender])[0],
        'Education': le_edu.transform([input_edu])[0],
        'Department': le_dept.transform([input_dept])[0],
        'Experience': input_exp,
        'Working_Hours': input_hours,
        'Performance_Score': input_perf,
        'Projects_Completed': input_projects
    }])
    
    user_scaled = scaler.transform(user_data)
    
    if st.button("Calculate Salary Metrics"):
        # Predictions
        pred_lr = lr.predict(user_scaled)[0]
        pred_rf = rf.predict(user_scaled)[0]
        pred_class_idx = log_reg.predict(user_scaled)[0]
        classes = ['Low Salary (<$60k)', 'Medium Salary ($60k-$110k)', 'High Salary (>$110k)']
        
        # Display Results
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Linear Regression Prediction", value=f"${pred_lr:,.2f}")
        c2.metric(label="Random Forest Prediction", value=f"${pred_rf:,.2f}")
        c3.metric(label="Classification Category", value=classes[pred_class_idx])

with tab3:
    st.subheader("Model Performance Summary")
    
    # Calculate metrics
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_rf = rf.predict(X_test_scaled)
    y_pred_log = log_reg.predict(X_test_scaled)
    
    metrics_data = {
        "Metric": ["MAE", "MSE", "R² Score / Accuracy"],
        "Linear Regression (Reg)": [f"{mean_absolute_error(y_test_reg, y_pred_lr):,.2f}", f"{mean_squared_error(y_test_reg, y_pred_lr):,.2f}", f"{r2_score(y_test_reg, y_pred_lr):.4f}"],
        "Random Forest (Reg)": [f"{mean_absolute_error(y_test_reg, y_pred_rf):,.2f}", f"{mean_squared_error(y_test_reg, y_pred_rf):,.2f}", f"{r2_score(y_test_reg, y_pred_rf):.4f}"],
        "Logistic Regression (Clf)": ["N/A", "N/A", f"{accuracy_score(y_test_clf, y_pred_log)*100:.2f}%"]
    }
    st.table(pd.DataFrame(metrics_data))
