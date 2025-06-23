import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset directly from GitHub folder
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

# Train model inside the app
@st.cache_resource
def train_model(df):
    df = df.dropna(subset=["price"])  # Drop missing target values
    df = df.drop(columns=["description", "name"])  # Drop unused text fields

    X = df.drop(columns=["price"])
    y = df["price"]

    num_features = ["year", "mileage", "cylinders", "doors"]
    cat_features = [col for col in X.columns if col not in num_features]

    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

# Load and train
df = load_data()
model = train_model(df)

# ------------------ UI ------------------
st.set_page_config(page_title="Vehicle Price Predictor", layout="wide")
st.title("🚗 Vehicle Price Predictor")
st.markdown("Estimate a car's price based on its specifications")

st.write("### 📋 Enter Vehicle Details:")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=1980, max_value=2025, value=2022)
    mileage = st.number_input("Mileage", min_value=0.0, value=10000.0)
    cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12], index=3)
    doors = st.selectbox("Doors", [2, 3, 4], index=2)

with col2:
    make = st.selectbox("Make", sorted(df["make"].dropna().unique().tolist()))
    model_name = st.text_input("Model", "Corolla")
    trim = st.text_input("Trim", "Base")
    fuel = st.selectbox("Fuel", sorted(df["fuel"].dropna().unique().tolist()))

with col3:
    transmission = st.selectbox("Transmission", sorted(df["transmission"].dropna().unique().tolist()))
    drivetrain = st.selectbox("Drivetrain", sorted(df["drivetrain"].dropna().unique().tolist()))
    body = st.selectbox("Body Style", sorted(df["body"].dropna().unique().tolist()))
    engine = st.text_input("Engine", "2.0L I4")

exterior_color = st.text_input("Exterior Color", "Silver")
interior_color = st.text_input("Interior Color", "Black")

if st.button("💲 Predict Price"):
    input_data = pd.DataFrame([{
        "year": year,
        "mileage": mileage,
        "cylinders": cylinders,
        "doors": doors,
        "make": make,
        "model": model_name,
        "trim": trim,
        "fuel": fuel,
        "transmission": transmission,
        "drivetrain": drivetrain,
        "body": body,
        "engine": engine,
        "exterior_color": exterior_color,
        "interior_color": interior_color
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
