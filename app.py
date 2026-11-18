"""
The web app that users will see.

Run with:
    streamlit run app.py
"""

import os

import joblib
import pandas as pd
import streamlit as st

MODEL_FOLDER = "models"

st.set_page_config(page_title="Price Predictor", layout="centered")

st.title("House and Car Price Predictor")
st.write("Choose what you want to predict, enter the details, and get an estimated price.")


@st.cache_resource
def load_house_model():
    """Load the trained house model once and reuse it (faster app)."""
    return joblib.load(os.path.join(MODEL_FOLDER, "house_model.pkl"))


@st.cache_resource
def load_car_model():
    """Load the trained car model, its column list and the brand list once."""
    model = joblib.load(os.path.join(MODEL_FOLDER, "car_model.pkl"))
    columns = joblib.load(os.path.join(MODEL_FOLDER, "car_columns.pkl"))
    brands = joblib.load(os.path.join(MODEL_FOLDER, "car_brands.pkl"))
    return model, columns, brands


def house_page():
    st.header("House Price Prediction")
    st.caption("Prices are in US dollars, matching the training data.")

    income = st.number_input("Average income in the area ($)", 20000, 120000, 68000, step=1000)
    age = st.slider("Average age of houses (years)", 2.5, 9.5, 6.0, 0.5)
    rooms = st.slider("Average number of rooms", 3.0, 11.0, 7.0, 0.5)
    bedrooms = st.slider("Average number of bedrooms", 2.0, 7.0, 4.0, 0.5)
    population = st.number_input("Area population", 5000, 70000, 36000, step=500)

    if st.button("Predict House Price", type="primary"):
        model = load_house_model()

        # put the user's values in a table with the same column names as training
        new_house = pd.DataFrame(
            [
                {
                    "Avg. Area Income": income,
                    "Avg. Area House Age": age,
                    "Avg. Area Number of Rooms": rooms,
                    "Avg. Area Number of Bedrooms": bedrooms,
                    "Area Population": population,
                }
            ]
        )

        price = max(float(model.predict(new_house)[0]), 1000)  # never show a silly price
        st.success(f"Estimated house price: ${price:,.0f}")


def car_page():
    st.header("Car Price Prediction")
    st.caption("Prices are in Indian rupees, matching the training data.")

    model, columns, brands = load_car_model()

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", brands)
        vehicle_age = st.slider("Vehicle age (years)", 0, 29, 6)
        km_driven = st.number_input("Kilometres driven", 100, 500000, 50000, step=1000)
        seats = st.selectbox("Number of seats", [2, 4, 5, 6, 7, 8, 9])
    with col2:
        fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        seller = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        mileage = st.number_input("Mileage (km per litre)", 4.0, 34.0, 19.7, step=0.1)
        engine = st.number_input("Engine capacity (cc)", 793, 6600, 1200, step=50)
        max_power = st.number_input("Max power (bhp)", 38.0, 630.0, 88.5, step=0.5)

    if st.button("Predict Car Price", type="primary"):
        new_car = pd.DataFrame(
            [
                {
                    "brand": brand,
                    "vehicle_age": vehicle_age,
                    "km_driven": km_driven,
                    "seller_type": seller,
                    "fuel_type": fuel,
                    "transmission_type": transmission,
                    "mileage": mileage,
                    "engine": engine,
                    "max_power": max_power,
                    "seats": seats,
                }
            ]
        )

        # turn text columns into 1/0 columns exactly like training did,
        # line them up with the model's columns, and make everything a plain number
        new_car = pd.get_dummies(new_car).reindex(columns=columns, fill_value=0).astype(float)

        price = max(float(model.predict(new_car)[0]), 10000)  # never show a silly price
        st.success(f"Estimated selling price: {price / 100000:.2f} lakhs (Rs {price:,.0f})")


def main():
    house_model_path = os.path.join(MODEL_FOLDER, "house_model.pkl")
    car_model_path = os.path.join(MODEL_FOLDER, "car_model.pkl")

    if not (os.path.exists(house_model_path) and os.path.exists(car_model_path)):
        st.error("Model files not found. First run:  python train.py")
        return

    with st.sidebar:
        st.title("Menu")
        choice = st.radio("Predict price of:", ("House", "Car"))
        st.caption("BCA Final Year Project")

    with st.expander("How does this work?"):
        st.write(
            "1. The app was trained on thousands of past sale records (details + actual price)."
        )
        st.write(
            "2. Linear Regression found the best formula that connects the details to the price."
        )
        st.write("3. When you enter new details, the app applies that formula to estimate the price.")
        st.write("4. Accuracy (R2 score) is printed when you run train.py. Closer to 1.0 is better.")

    if choice == "House":
        house_page()
    else:
        car_page()


main()
