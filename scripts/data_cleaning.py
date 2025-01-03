import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    filename="rossmann_sales.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

try:
    # Load data with explicit dtype where necessary
    train = pd.read_csv("train.csv", dtype={"StateHoliday": "str"})
    test = pd.read_csv("test.csv", dtype={"StateHoliday": "str"})
    store = pd.read_csv("store.csv")
    logging.info("Data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise

# Merge train and store data
try:
    train = train.merge(store, on="Store", how="left")
    test = test.merge(store, on="Store", how="left")
    logging.info("Train and test data merged with store data.")
except Exception as e:
    logging.error(f"Error merging data: {e}")
    raise

# Fill missing values
try:
    train["CompetitionDistance"] = train["CompetitionDistance"].fillna(
        train["CompetitionDistance"].max()
    )
    train["CompetitionOpenSinceMonth"] = train["CompetitionOpenSinceMonth"].fillna(0)
    train["CompetitionOpenSinceYear"] = train["CompetitionOpenSinceYear"].fillna(0)
    train["Promo2SinceYear"] = train["Promo2SinceYear"].fillna(0)
    train["Promo2SinceWeek"] = train["Promo2SinceWeek"].fillna(0)
    train["PromoInterval"] = train["PromoInterval"].fillna("None")

    test["CompetitionDistance"] = test["CompetitionDistance"].fillna(
        train["CompetitionDistance"].max()
    )
    test["CompetitionOpenSinceMonth"] = test["CompetitionOpenSinceMonth"].fillna(0)
    test["CompetitionOpenSinceYear"] = test["CompetitionOpenSinceYear"].fillna(0)
    test["Promo2SinceYear"] = test["Promo2SinceYear"].fillna(0)
    test["Promo2SinceWeek"] = test["Promo2SinceWeek"].fillna(0)
    test["PromoInterval"] = test["PromoInterval"].fillna("None")

    logging.info("Missing values filled.")
except Exception as e:
    logging.error(f"Error filling missing values: {e}")
    raise

# Feature engineering
try:
    # Add year and month columns for calculations
    train["Year"] = pd.to_datetime(train["Date"]).dt.year
    train["Month"] = pd.to_datetime(train["Date"]).dt.month
    train["WeekOfYear"] = pd.to_datetime(train["Date"]).dt.isocalendar().week

    test["Year"] = pd.to_datetime(test["Date"]).dt.year
    test["Month"] = pd.to_datetime(test["Date"]).dt.month
    test["WeekOfYear"] = pd.to_datetime(test["Date"]).dt.isocalendar().week

    # Competition active flag
    train["CompetitionActive"] = (
        (train["CompetitionOpenSinceYear"] < train["Year"])
        | (
            (train["CompetitionOpenSinceYear"] == train["Year"])
            & (train["CompetitionOpenSinceMonth"] <= train["Month"])
        )
    ).astype(int)

    test["CompetitionActive"] = (
        (test["CompetitionOpenSinceYear"] < test["Year"])
        | (
            (test["CompetitionOpenSinceYear"] == test["Year"])
            & (test["CompetitionOpenSinceMonth"] <= test["Month"])
        )
    ).astype(int)

    # Promo2 active flag
    train["Promo2Active"] = (
        (train["Promo2SinceYear"] < train["Year"])
        | (
            (train["Promo2SinceYear"] == train["Year"])
            & (train["Promo2SinceWeek"] <= train["WeekOfYear"])
        )
    ).astype(int)

    test["Promo2Active"] = (
        (test["Promo2SinceYear"] < test["Year"])
        | (
            (test["Promo2SinceYear"] == test["Year"])
            & (test["Promo2SinceWeek"] <= test["WeekOfYear"])
        )
    ).astype(int)

    logging.info("Feature engineering completed.")
except Exception as e:
    logging.error(f"Error during feature engineering: {e}")
    raise

# Save cleaned data for further processing
try:
    train.to_csv("train_cleaned.csv", index=False)
    test.to_csv("test_cleaned.csv", index=False)
    logging.info("Cleaned data saved successfully.")
except Exception as e:
    logging.error(f"Error saving cleaned data: {e}")
    raise
