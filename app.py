import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
model = pickle.load(open("src/model.pkl", "rb"))

# Load any other dependencies
locations = ['Downtown', 'Marina', 'JLT', 'Business Bay', 'Jumeirah']

# App title
st.title("Dubai House Price Prediction")

# Input features
st.sidebar.header("Enter House Details")
location = st.sidebar.selectbox("Location", locations)
size = st.sidebar.slider("Size (sqft)", 500, 10000, step=100)
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, step=1)
bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 10, step=1)
year = st.sidebar.number_input("Year Built", min_value=1900, max_value=2025, value=2020)

# Predict button
if st.sidebar.button("Predict"):
    input_data = pd.DataFrame({
        "location": [location],
        "size": [size],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "year": [year]
    })
    
    # Predict price
    prediction = model.predict(input_data)[0]
    st.write(f"Predicted House Price: AED {round(prediction, 2)}")

