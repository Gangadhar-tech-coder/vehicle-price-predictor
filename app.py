import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64

def set_background_with_overlay(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.9)), url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Use this in your app
set_background_with_overlay("image3.jpg")



# Page config
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load model
model = joblib.load("vehicle_price_model.pkl")

st.markdown("<h1 style='text-align: center;'>🚗 Vehicle Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enter vehicle features to estimate its price</p>", unsafe_allow_html=True)
st.write("---")

# --- Section: Input form ---
with st.form("input_form"):
    st.subheader("🔧 Vehicle Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.number_input("Manufacturing Year", min_value=1980, max_value=2025, step=1, value=2022)
        mileage = st.number_input("Mileage (in miles)", min_value=0.0, value=5000.0)
        cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12], index=3)
        doors = st.selectbox("Doors", [2, 3, 4], index=2)

    with col2:
        make = st.selectbox("Make", ["Ford", "Toyota", "BMW", "Jeep", "RAM", "Chevrolet", "GMC", "Other"])
        model_name = st.text_input("Model", placeholder="e.g. Corolla, Yukon XL")
        trim = st.text_input("Trim Level", placeholder="e.g. Limited, Base")

    with col3:
        fuel = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid", "Other"])
        transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT", "Other"])
        drivetrain = st.selectbox("Drivetrain", ["All-wheel Drive", "Four-wheel Drive", "Front-wheel Drive", "Rear-wheel Drive", "Other"])
        body = st.selectbox("Body Style", ["SUV", "Sedan", "Pickup Truck", "Hatchback", "Coupe", "Other"])

    st.markdown("### 🎨 Appearance Details")
    col4, col5 = st.columns(2)
    with col4:
        exterior_color = st.text_input("Exterior Color", placeholder="e.g. Silver, Red")
    with col5:
        interior_color = st.text_input("Interior Color", placeholder="e.g. Black, Beige")

    st.markdown("### ⚙️ Engine Details")
    engine = st.text_input("Engine Description", placeholder="e.g. 6.2L V8 Turbocharged")

    submitted = st.form_submit_button("💲 Predict Price")

# --- Section: Output ---
if submitted:
    input_data = {
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
        "exterior_color": exterior_color,
        "interior_color": interior_color,
        "engine": engine
    }

    input_df = pd.DataFrame([input_data])

    try:
        prediction = model.predict(input_df)[0]

        st.success("✅ Prediction Complete!")
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border: 1px solid #ddd; text-align: center;'>
            <h2>💰 Estimated Price: <span style='color: #2ecc71;'>${prediction:,.2f}</span></h2>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")

# --- Footer ---
st.markdown("""
<hr>
<p style='text-align: center; font-size: 0.9em; color: gray;'>
    Made with ❤️ using Machine Learning | © 2025 VehiclePredictor.ai
</p>
""", unsafe_allow_html=True)
