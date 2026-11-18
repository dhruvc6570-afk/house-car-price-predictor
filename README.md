# House and Car Price Predictor

A simple machine learning web app built as a BCA final year project.
It predicts **house prices** (US housing data) and **car resale prices** (Indian CarDekho data)
using **Linear Regression**, the easiest machine learning algorithm to explain.

## How it works (the one-paragraph explanation)

We show the model thousands of past sale records: the details of each sale (area, rooms,
year, kilometres, etc.) and the final price. Linear Regression studies these examples and
finds the formula that best connects the details to the price. When a user enters new
details in the app, the model applies that formula and returns an estimated price.
Accuracy is measured with the R2 score (1.0 = perfect, 0 = useless); it is printed when training.

## Project structure

```
price_predictor/
├── data/                     <- CSV files go here (sample or Kaggle)
├── models/                   <- trained model files are saved here automatically
├── generate_sample_data.py   <- creates quick-start sample data (same columns as Kaggle)
├── train.py                  <- trains both models and prints their accuracy
├── app.py                    <- the Streamlit web app (House / Car toggle)
└── requirements.txt          <- list of libraries to install
```

## Setup (one time)

```
pip install -r requirements.txt
```

## Step 1: Get the data (two options)

**Option A - Quick start (no download needed):**

```
python generate_sample_data.py
```

**Option B - Real data from Kaggle (recommended for the final report):**

1. Create a free account at https://www.kaggle.com
2. House data: open https://www.kaggle.com/datasets/vedavyasv/usa-housing
   - click **Download**, unzip it
   - rename the file `USA_Housing.csv` to `house_data.csv`
   - move it into the `data/` folder
3. Car data: the used-car price dataset with columns
   `car_name, brand, model, vehicle_age, km_driven, seller_type, fuel_type,
   transmission_type, mileage, engine, max_power, seats, selling_price`
   (15,411 rows, prices in rupees)
   - rename it to `car_data.csv` and move it into the `data/` folder (already done)

The column names of the sample data and the real Kaggle data are identical,
so you can switch between them freely. After changing the data, just train again.

## Step 2: Train the models

```
python train.py
```

This prints the accuracy (R2 score) of both models and saves them into `models/`.

## Step 3: Run the app

```
streamlit run app.py
```

The app opens in your browser at http://localhost:8501

## What to say in the viva (short version)

- **Problem:** people have no quick way to estimate a fair price for a house or a used car.
- **Solution:** an app that learns from past sales and predicts prices from details.
- **Algorithm:** Linear Regression - it fits the straight line (formula) that best
  connects input details to price.
- **Data split:** 80% of rows train the model, 20% test it on unseen rows.
- **Accuracy metric:** R2 score - how much of the price variation the model explains.
- **Text to numbers:** columns like `brand`, `fuel_type`, `seller_type`,
  `transmission_type` are converted using one-hot encoding (pandas `get_dummies`),
  because the model only understands numbers.
- **Data cleaning:** removed 14 rows with impossible kilometre readings (over 5 lakh km).
- **Saving the model:** `joblib` stores the trained model in a file so the app
  does not need to train again every time it starts.
