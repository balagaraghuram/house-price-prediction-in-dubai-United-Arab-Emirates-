# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ============================
# Model Training and Saving
# ============================

def train_and_save_model(model_path):
    """
    Generates synthetic data, trains a RandomForestRegressor,
    and saves the trained model to the specified path.
    """
    try:
        # Synthetic Dataset Creation
        np.random.seed(42)  # For reproducibility
        num_samples = 1000

        # Define possible categories
        locations = ['Downtown', 'Marina', 'Jumeirah', 'Business Bay', 'Other']
        property_types = ['Apartment', 'Villa', 'Townhouse', 'Studio']

        # Generate synthetic features
        data = pd.DataFrame({
            'Location': np.random.choice(locations, size=num_samples),
            'Property_Type': np.random.choice(property_types, size=num_samples),
            'Bedrooms': np.random.randint(1, 6, size=num_samples),
            'Bathrooms': np.random.randint(1, 5, size=num_samples),
            'Square_Footage': np.random.randint(500, 5000, size=num_samples),
            'Property_Age': np.random.randint(0, 50, size=num_samples)
        })

        # Generate synthetic target variable (House Price)
        # For simplicity, let's assume a base price plus some influence from features
        base_price = 50000
        price = (
            base_price +
            data['Bedrooms'] * 30000 +
            data['Bathrooms'] * 20000 +
            data['Square_Footage'] * 100 +
            (50 - data['Property_Age']) * 500  # Newer properties are more valuable
        )

        data['Price'] = price + np.random.normal(0, 10000, size=num_samples)  # Adding some noise

        # Encode categorical variables
        location_map = {'Downtown': 0, 'Marina': 1, 'Jumeirah': 2, 'Business Bay': 3, 'Other': 4}
        property_type_map = {'Apartment': 0, 'Villa': 1, 'Townhouse': 2, 'Studio': 3}

        data['Location'] = data['Location'].map(location_map)
        data['Property_Type'] = data['Property_Type'].map(property_type_map)

        # Define feature matrix X and target vector y
        X = data[['Location', 'Property_Type', 'Bedrooms', 'Bathrooms', 'Square_Footage', 'Property_Age']]
        y = data['Price']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, model_path)
        st.success(f"Model trained and saved successfully at '{model_path}'.")
    except Exception as e:
        st.error("An error occurred during model training and saving.")
        st.text(traceback.format_exc())

# ============================
# Streamlit Application
# ============================

# Set the title of the app
st.title("🏠 House Price Prediction in Dubai, UAE")

st.write("""
This application predicts the **House Price** in Dubai based on various features.
""")

# Define the path for the model
MODEL_PATH = 'model.pkl'

# Check if the model exists; if not, train and save it
if not os.path.isfile(MODEL_PATH):
    st.warning(f"Model file '{MODEL_PATH}' not found. Training a new model...")
    train_and_save_model(MODEL_PATH)

# Load the pre-trained model
@st.cache_resource
def load_model(model_path):
    """
    Loads the trained model from the specified path.
    """
    try:
        loaded_model = joblib.load(model_path)
        return loaded_model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.text(traceback.format_exc())
        return None

model = load_model(MODEL_PATH)

# Function to get user input
def get_user_input():
    st.sidebar.header('Specify Input Parameters')

    # Define the input options
    locations = ('Downtown', 'Marina', 'Jumeirah', 'Business Bay', 'Other')
    property_types = ('Apartment', 'Villa', 'Townhouse', 'Studio')

    # Sidebar inputs
    location = st.sidebar.selectbox('Location', locations)
    property_type = st.sidebar.selectbox('Property Type', property_types)
    bedrooms = st.sidebar.number_input('Number of Bedrooms', min_value=0, max_value=10, value=3, step=1)
    bathrooms = st.sidebar.number_input('Number of Bathrooms', min_value=0, max_value=10, value=2, step=1)
    sqft = st.sidebar.number_input('Square Footage (sqft)', min_value=100, max_value=10000, value=1500, step=100)
    age = st.sidebar.number_input('Property Age (years)', min_value=0, max_value=100, value=5, step=1)

    # Encode categorical variables
    location_map = {'Downtown': 0, 'Marina': 1, 'Jumeirah': 2, 'Business Bay': 3, 'Other': 4}
    property_type_map = {'Apartment': 0, 'Villa': 1, 'Townhouse': 2, 'Studio': 3}

    location_encoded = location_map.get(location, 4)
    property_type_encoded = property_type_map.get(property_type, 3)

    # Create a DataFrame for the input features
    user_data = {
        'Location': location_encoded,
        'Property_Type': property_type_encoded,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Square_Footage': sqft,
        'Property_Age': age
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
input_df = get_user_input()

# Display user input
st.subheader('📝 User Input Parameters')
st.write(input_df)

# Prediction
if st.button('🔮 Predict'):
    if model:
        try:
            prediction = model.predict(input_df)
            st.success(f'**Estimated House Price:** AED {prediction[0]:,.2f}')
        except Exception as e:
            st.error("Error in prediction:")
            st.text(traceback.format_exc())
    else:
        st.error("Model is not loaded properly.")

# Additional Information
st.sidebar.markdown("""
---
**Developed by:** Your Name  
**Contact:** your.email@example.com
""")
