import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model
# Make sure 'breast_cancer_model.pkl' is in the same directory as this script
model = joblib.load('breast_cancer_model.pkl')

# App title
st.title("Breast Cancer Prediction")
st.write("Enter the details below to predict if the tumor is benign or malignant.")

# List of features
features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Create input fields for each feature
input_values = []
for feature in features:
    value = st.number_input(f"Enter {feature}", min_value=0.0, step=0.1, key=feature)
    input_values.append(value)

# Predict button
if st.button("Predict"):
    # Convert input values to a NumPy array
    input_data = np.array([input_values])
    
    # Make a prediction
    try:
        prediction = model.predict(input_data)
        
        # Display the result
        if prediction[0] == 0:
            st.success("The tumor is predicted to be **Benign**.")
        else:
            st.error("The tumor is predicted to be **Malignant**.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
