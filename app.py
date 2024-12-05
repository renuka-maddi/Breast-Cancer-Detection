import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('breast_cancer_model.pkl')

# Title
st.title("Breast Cancer Prediction")
st.write("Enter the details below to predict if the tumor is benign or malignant.")

# Input fields
mean_radius = st.number_input("Mean Radius", min_value=0.0, max_value=50.0, step=0.1)
mean_texture = st.number_input("Mean Texture", min_value=0.0, max_value=50.0, step=0.1)
mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, max_value=200.0, step=0.1)
mean_area = st.number_input("Mean Area", min_value=0.0, max_value=2500.0, step=0.1)
