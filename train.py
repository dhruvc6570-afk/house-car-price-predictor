"""
Trains two Linear Regression models and saves them:

1. models/house_model.pkl -> predicts house price from area details
2. models/car_model.pkl   -> predicts car selling price from car details

Run this AFTER the CSV files are inside the data/ folder:
    python train.py
"""

import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

DATA_FOLDER = "data"
MODEL_FOLDER = "models"


def train_house_model():
    print("=" * 50)
    print("TRAINING HOUSE PRICE MODEL")
    print("=" * 50)

    df = pd.read_csv(os.path.join(DATA_FOLDER, "house_data.csv"))

    # The real Kaggle file has an "Address" column (just text) -> remove it
    if "Address" in df.columns:
        df = df.drop(columns=["Address"])

    print("Missing values in data:", int(df.isnull().sum().sum()))

    # X = the details we use to predict, y = the price we want to predict
    X = df.drop(columns=["Price"])
    y = df["Price"]

    # 80% of rows teach the model, 20% are kept aside to test it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)  # this is where the learning happens

    accuracy = model.score(X_test, y_test)  # R2 score: 1.0 = perfect, 0 = useless
    print(f"Rows used: {len(df)}")
    print(f"House model accuracy (R2 score): {accuracy:.3f}  (1.0 = perfect)")

    # Show what the model learned: how much each feature adds to the price
    print("Learned effect of each feature on price:")
    for name, coef in zip(X.columns, model.coef_):
        print(f"   {name}: {coef:,.0f}")

    joblib.dump(model, os.path.join(MODEL_FOLDER, "house_model.pkl"))
    print("Saved -> models/house_model.pkl")
    print()


def train_car_model():
    print("=" * 50)
    print("TRAINING CAR PRICE MODEL")
    print("=" * 50)

    df = pd.read_csv(os.path.join(DATA_FOLDER, "car_data.csv"))

    # car_name and model are just labels with hundreds of unique values;
    # the brand already tells the model most of what it needs -> drop them
    # (only drop a column if it exists, so sample data also works)
    for col in ["Unnamed: 0", "car_name", "model"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # data cleaning: a few cars show impossible kilometre readings
    # (e.g. 38 lakh km). Cars almost never cross 5 lakh km -> remove them
    before = len(df)
    df = df[df["km_driven"] < 500000]
    print(f"Removed {before - len(df)} rows with impossible km readings")

    print("Missing values in data:", int(df.isnull().sum().sum()))

    X = df.drop(columns=["selling_price"])
    y = df["selling_price"]  # price in rupees

    # Text columns (brand, fuel_type, seller_type, transmission_type) must
    # become numbers. get_dummies makes one new 1/0 column per category,
    # e.g. fuel_type = "Diesel" becomes a column fuel_type_Diesel = 1
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Rows used: {len(df)}")
    print(f"Car model accuracy (R2 score): {accuracy:.3f}  (1.0 = perfect)")

    # Save the model AND the exact column list, so the app can prepare
    # new inputs in exactly the same way as during training
    joblib.dump(model, os.path.join(MODEL_FOLDER, "car_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_FOLDER, "car_columns.pkl"))
    # save the brand list so the app can offer the same brands as the data
    joblib.dump(sorted(df["brand"].unique().tolist()), os.path.join(MODEL_FOLDER, "car_brands.pkl"))
    print("Saved -> models/car_model.pkl")
    print()


def main():
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    house_path = os.path.join(DATA_FOLDER, "house_data.csv")
    car_path = os.path.join(DATA_FOLDER, "car_data.csv")

    if not (os.path.exists(house_path) and os.path.exists(car_path)):
        print("CSV files not found inside the data/ folder!")
        print("Option 1 (quick start):  python generate_sample_data.py")
        print("Option 2 (real data):    download from Kaggle, steps in README.md")
        return

    train_house_model()
    train_car_model()

    print("DONE! Now run the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
