import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("xgb_best_model.pkl")

st.title("Heart Disease Prediction App")

st.markdown("### Please enter your medical details below:")

# Add descriptions and inputs
age = st.number_input("Age", 1, 120, help="Enter your age in years")

sex = st.radio("Sex", ["Male", "Female"], help="Biological sex of the person")

cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
    help="""
0: Typical Angina  
1: Atypical Angina  
2: Non-Anginal Pain  
3: Asymptomatic
""")

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, help="Blood pressure during rest (normal is around 120)")

chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, help="Normal range is below 200 mg/dL")

fbs = st.radio("Fasting Blood Sugar > 120 mg/dL?", [0, 1],
    help="1: Yes, 0: No")

restecg = st.selectbox("Resting ECG Results", [0, 1, 2],
    help="""
0: Normal  
1: ST-T wave abnormality  
2: Left ventricular hypertrophy
""")

thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, help="Typical values range from 100–200 during exercise")

exang = st.radio("Exercise Induced Angina", [0, 1],
    help="1: Yes, 0: No")

oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, step=0.1,
    help="ST depression induced by exercise relative to rest")

slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2],
    help="""
0: Upsloping  
1: Flat  
2: Downsloping
""")

if st.button("Predict"):
    # Format input
    input_data = np.array([[
        age,
        1 if sex == "Male" else 0,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ You are likely to have heart disease.")
    else:
        st.success("✅ You are unlikely to have heart disease.")
