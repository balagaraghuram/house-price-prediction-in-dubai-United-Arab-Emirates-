import streamlit as st
import pandas as pd
import numpy as np
import pickle  # Reverted to pickle
import os
import traceback  # For detailed error messages

# Set the title of the app
st.title("🏠 House Price Prediction in Dubai, UAE")

st.write("""
This application predicts the **House Price** in Dubai based on various features.
""")

# Load the pre-trained model
@st.cache_resource  # Ensure you're using Streamlit version that supports cache_resource
def load_model():
    model_path = 'model.pkl'
    if not os.path.isfile(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure it is uploaded to the repository.")
        return None
    try:
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)  # Use pickle to load the model
        return loaded_model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.text(traceback.format_exc())  # Display the full traceback
        return None

model = load_model()

# Function to get user input (remains unchanged)
def get_user_input():
    st.sidebar.header('Specify Input Parameters')

    # Example input fields - replace these with features relevant to your model
    # Adjust or add/remove features as per your model's requirements

    location = st.sidebar.selectbox('Location', ('Downtown', 'Marina', 'Jumeirah', 'Business Bay', 'Other'))
    property_type = st.sidebar.selectbox('Property Type', ('Apartment', 'Villa', 'Townhouse', 'Studio'))
    bedrooms = st.sidebar.number_input('Number of Bedrooms', min_value=0, max_value=10, value=3)
    bathrooms = st.sidebar.number_input('Number of Bathrooms', min_value=0, max_value=10, value=2)
    sqft = st.sidebar.number_input('Square Footage (sqft)', min_value=100, max_value=10000, value=1500)
    age = st.sidebar.number_input('Property Age (years)', min_value=0, max_value=100, value=5)

    # Encode categorical variables as needed
    # This should match the preprocessing done during model training
    location_map = {'Downtown': 0, 'Marina': 1, 'Jumeirah': 2, 'Business Bay': 3, 'Other': 4}
    property_type_map = {'Apartment': 0, 'Villa': 1, 'Townhouse': 2, 'Studio': 3}

    location_encoded = location_map.get(location, 4)
    property_type_encoded = property_type_map.get(property_type, 3)

    # Create a dictionary for the input features
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
st.subheader('User Input Parameters')
st.write(input_df)

# Prediction
if st.button('Predict'):
    if model:
        try:
            prediction = model.predict(input_df)
            st.success(f'Estimated House Price: AED {prediction[0]:,.2f}')
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            st.text(traceback.format_exc())  # Display the full traceback
    else:
        st.error("Model is not loaded properly.")

# Additional Information
st.sidebar.markdown("""
---
**Developed by:** Balaga Raghuram

 
""")
