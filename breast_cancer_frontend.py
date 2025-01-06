import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset and train the model
breast_cancer_dataset = load_breast_cancer()
X = breast_cancer_dataset.data
Y = breast_cancer_dataset.target

model = LogisticRegression(max_iter=5000)
model.fit(X, Y)

# Streamlit Frontend
st.title("Breast Cancer Detection")

st.sidebar.header("Input Features")
input_data = []

# Generate input fields for each feature
for feature in breast_cancer_dataset.feature_names:
    value = st.sidebar.number_input(feature, value=0.0)
    input_data.append(value)

if st.sidebar.button("Predict"):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_as_numpy_array)
    
    if prediction[0] == 0:
        st.success("The Breast Cancer is Malignant")
    else:
        st.success("The Breast Cancer is Benign")
