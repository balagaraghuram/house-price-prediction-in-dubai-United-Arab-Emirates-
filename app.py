import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Title of the app
st.title("Dubai House Price Prediction App 🏠")
st.write("Predict house prices in Dubai based on property details like location, size, and more!")

# Sidebar for user inputs
st.sidebar.header("Enter Property Details")

# User inputs
location = st.sidebar.selectbox(
    "Location",
    ["Downtown", "Marina", "JLT", "Business Bay", "Jumeirah"]
)
size = st.sidebar.slider("Property Size (sqft)", 500, 10000, step=100, value=1500)
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, step=1, value=3)
bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 10, step=1, value=2)
year_built = st.sidebar.number_input("Year Built", min_value=1900, max_value=2025, value=2015)

# Check if the model exists
try:
    # Load the pre-trained model if available
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("Loaded pre-trained model successfully!")
except FileNotFoundError:
    # If no pre-trained model is found, train a new model with synthetic data
    st.warning("Pre-trained model not found. Training a new model...")

    # Generate synthetic dataset
    data = {
        "location": ["Downtown", "Marina", "JLT", "Business Bay", "Jumeirah"] * 100,
        "size": np.random.randint(500, 10000, 500),
        "bedrooms": np.random.randint(1, 10, 500),
        "bathrooms": np.random.randint(1, 10, 500),
        "year_built": np.random.randint(1990, 2025, 500),
        "price": np.random.randint(500000, 5000000, 500),
    }
    df = pd.DataFrame(data)

    # One-hot encode location
    df = pd.get_dummies(df, columns=["location"], drop_first=True)

    # Features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model trained and saved successfully!")

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.info(f"New model trained. Mean Squared Error: {mse}")

# Prediction
if st.sidebar.button("Predict Price"):
    # Prepare input data for prediction
    input_data = {
        "size": [size],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "year_built": [year_built],
        "location_Marina": [1 if location == "Marina" else 0],
        "location_JLT": [1 if location == "JLT" else 0],
        "location_Business Bay": [1 if location == "Business Bay" else 0],
        "location_Jumeirah": [1 if location == "Jumeirah" else 0],
    }
    input_df = pd.DataFrame(input_data)

    # Predict the house price
    predicted_price = model.predict(input_df)[0]
    st.write(f"### Predicted House Price: AED {round(predicted_price, 2):,}")

# Footer
st.write("### Powered by Machine Learning | Built with Streamlit 🚀")
