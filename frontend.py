import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))
#scaler = pickle.load(open('scaler.pkl', 'rb'))
# Load the scaler
scaler = StandardScaler()
scaler.fit([[17.90,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]])  # Use sample training data

st.title("Breast Cancer Prediction")
st.write("Enter the required parameters below to predict whether the tumor is Benign or Malignant.")

# Input fields
features = []
feature_names = ['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
                 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
                 'Radius Error', 'Texture Error', 'Perimeter Error', 'Area Error', 'Smoothness Error',
                 'Compactness Error', 'Concavity Error', 'Concave Points Error', 'Symmetry Error', 'Fractal Dimension Error',
                 'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness',
                 'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension']

for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0, format="%f")
    features.append(value)

# Predict button
if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    result = "Benign: Non-Cancerous" if prediction[0] == 1 else "Malignant: Cancerous"
    st.success(f"Prediction: {result}")
