"""
Creates sample CSV files inside the data/ folder so the project can run
immediately, without downloading anything.

The column names are EXACTLY the same as the real Kaggle datasets, so later
you can replace these files with real data and nothing else needs to change.

Real datasets this file imitates:
- House: USA_Housing  -> https://www.kaggle.com/datasets/vedavyasv/usa-housing
- Car  : used-car price dataset (same columns as the real car_data.csv)
"""

import os

import numpy as np
import pandas as pd

np.random.seed(42)  # same random numbers every run -> same data every run

DATA_FOLDER = "data"


def make_house_data(rows=5000):
    """Creates house rows. Price = effect of each feature + small noise."""
    income = np.random.normal(68000, 15000, rows).clip(20000, 120000)        # average income of people in the area ($)
    age = np.random.normal(6.0, 1.2, rows).clip(2.5, 9.5).round(1)           # average age of houses (years)
    rooms = np.random.normal(7.0, 1.0, rows).clip(3.2, 10.8).round(2)        # average number of rooms
    bedrooms = np.random.normal(4.0, 0.8, rows).clip(2.0, 7.0).round(1)      # average number of bedrooms
    population = np.random.normal(36000, 8700, rows).clip(5000, 69600).round(0)  # population of the area

    # The pattern the model must learn: each feature adds its share to the price
    price = (
        -2630000
        + 21.5 * income
        + 165500 * age
        + 120000 * rooms
        + 2200 * bedrooms
        + 15.4 * population
        + np.random.normal(0, 100000, rows)  # noise = things we cannot measure
    ).clip(50000).round(0)

    return pd.DataFrame(
        {
            "Avg. Area Income": income,
            "Avg. Area House Age": age,
            "Avg. Area Number of Rooms": rooms,
            "Avg. Area Number of Bedrooms": bedrooms,
            "Area Population": population,
            "Price": price,
        }
    )


def make_car_data(rows=3000):
    """Creates car rows. Selling price = base value + brand premium - depreciation - usage."""
    brand_choices = [
        "Maruti", "Maruti", "Maruti", "Hyundai", "Hyundai", "Honda", "Toyota",
        "Tata", "Mahindra", "Ford", "Renault", "Volkswagen", "BMW", "Audi",
    ]
    premiums = {  # how much extra a brand adds to the price (rupees)
        "Maruti": 0, "Hyundai": 20000, "Honda": 40000, "Toyota": 100000,
        "Tata": 10000, "Mahindra": 30000, "Ford": 30000, "Renault": 0,
        "Volkswagen": 60000, "BMW": 1500000, "Audi": 1600000,
    }
    brand = np.random.choice(brand_choices, rows)
    vehicle_age = np.clip(np.random.gamma(2.5, 2.4, rows), 0, 29).round(0).astype(int)
    km_driven = (8000 * (vehicle_age + 1) * np.random.uniform(0.5, 2.0, rows)).clip(100, 500000).round(-2)
    seller_type = np.random.choice(["Individual", "Dealer", "Trustmark Dealer"], rows, p=[0.60, 0.30, 0.10])
    fuel_type = np.random.choice(["Petrol", "Diesel", "CNG", "LPG", "Electric"], rows, p=[0.55, 0.35, 0.04, 0.03, 0.03])
    transmission_type = np.random.choice(["Manual", "Automatic"], rows, p=[0.75, 0.25])
    mileage = np.random.normal(19.7, 4.0, rows).clip(4.0, 34.0).round(1)     # km per litre
    engine = np.random.normal(1486, 520, rows).clip(793, 3600).round(0)      # engine capacity in cc
    max_power = (engine * np.random.uniform(0.04, 0.09, rows)).clip(38.4, 630).round(1)  # power in bhp
    seats = np.random.choice([2, 4, 5, 6, 7, 8, 9], rows, p=[0.05, 0.05, 0.60, 0.03, 0.20, 0.04, 0.03])

    premium = np.array([premiums[b] for b in brand])
    selling_price = (
        250000
        + 6000 * max_power        # more powerful cars cost more
        + 150 * engine            # bigger engines cost more
        - 35000 * vehicle_age     # value drops every year
        - 0.8 * km_driven         # more kilometres -> lower value
        + premium                 # premium brands cost more
        - 3000 * (mileage - 19.7)
        + np.random.normal(0, 200000, rows)  # noise
    ).clip(40000, 40000000).round(-2)

    return pd.DataFrame(
        {
            "car_name": brand + " Model",
            "brand": brand,
            "model": "Model",
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "seller_type": seller_type,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats,
            "selling_price": selling_price,
        }
    )


def main():
    os.makedirs(DATA_FOLDER, exist_ok=True)

    make_house_data().to_csv(os.path.join(DATA_FOLDER, "house_data.csv"), index=False)
    make_car_data().to_csv(os.path.join(DATA_FOLDER, "car_data.csv"), index=False)

    print("Sample data created:")
    print(" - data/house_data.csv  (5000 rows)")
    print(" - data/car_data.csv    (3000 rows)")
    print("You can replace these later with the real Kaggle files (steps in README.md).")


if __name__ == "__main__":
    main()
