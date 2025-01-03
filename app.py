import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Title of the app
st.title("Dubai House Price Prediction App")

# Sidebar for user inputs
st.sidebar.header("Enter House Details")

# Input fields
location = st.sidebar.selectbox("Location", ["Downtown", "Marina", "JLT", "Business Bay", "Jumeirah"])
size = st.sidebar.slider("Size (sqft)", 500, 10000, step=100)
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, step=1)
bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 10, step=1)
year_built = st.sidebar.number_input("Year Built", min_value=1900, max_value=2025, value=2020)

# Check if the model exists
try:
    # Load pre-trained model if it exists
    model = pickle.load(open("model.pkl", "rb"))
    st.success("Model loaded successfully!")
except FileNotFoundError:
    # Train a new model if it doesn't exist
    st.warning("Model file not found. Training a new model...")

    # Generate a sample dataset
    data = {
        "location": ["Downtown", "Marina", "JLT", "Business Bay", "Jumeirah"] * 20,
        "size": np.random.randint(500, 10000, 100),
        "bedrooms": np.random.randint(1, 10, 100),
        "bathrooms": np.random.randint(1, 10, 100),
        "year_built": np.random.randint(1990, 2025, 100),
        "price": np.random.randint(500000, 5000000, 100),
    }
    df = pd.DataFrame(data)

    # Encode location as one-hot (example categorical feature)
    df = pd.get_dummies(df, columns=["location"], drop_first=True)

    # Features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.info(f"New model trained. MSE: {mse}")

    # Save the model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model saved successfully!")

# Prediction button
if st.sidebar.button("Predict"):
    # Prepare input data
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

    # Predict house price
    predicted_price = model.predict(input_df)[0]
    st.write(f"### Predicted House Price: AED {round(predicted_price, 2):,}")

# Footer
st.write("### Powered by Machine Learning | Built with Streamlit")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
except ImportError as e:
    st.error(f"Required library missing: {e}. Please install the dependencies using `pip install -r requirements.txt`.")
    st.stop()

