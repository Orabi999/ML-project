"""Train and export the house-price prediction pipeline."""
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = BASE_DIR / "data" / "house_prices.csv"
MODEL_DIR = PROJECT_DIR / "backend" / "models"
BACKEND_LOCATIONS = PROJECT_DIR / "backend" / "locations.json"
FRONTEND_LOCATIONS = PROJECT_DIR / "frontend" / "src" / "locations.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
FRONTEND_LOCATIONS.parent.mkdir(parents=True, exist_ok=True)

# Create a small demo dataset only when no real CSV is supplied.
if not DATA_PATH.exists():
    mock_data = pd.DataFrame({
        "Amount (in rupees)": ["50 Lac", "1.2 Cr", "75 Lac", "30 Lac", "2.5 Cr", "80 Lac", "45 Lac", "1.8 Cr"],
        "Carpet Area": ["1000 sqft", "1500 sqft", "1200 sqft", "800 sqft", "2200 sqft", "1300 sqft", "950 sqft", "1900 sqft"],
        "Super Area": ["1200 sqft", "1800 sqft", "1400 sqft", "950 sqft", "2500 sqft", "1500 sqft", "1100 sqft", "2100 sqft"],
        "Floor": ["2nd", "5th", "Ground", "1st", "10th", "3rd", "4th", "8th"],
        "Bathroom": [2, 3, 2, 1, 4, 2, 2, 3],
        "Balcony": [1, 2, 1, 1, 3, 1, 1, 2],
        "location": ["Downtown", "Suburbs", "Downtown", "Uptown", "Suburbs", "Downtown", "Uptown", "Suburbs"],
        "Furnishing": ["Furnished", "Semi-Furnished", "Unfurnished", "Furnished", "Semi-Furnished", "Furnished", "Unfurnished", "Semi-Furnished"],
        "Transaction": ["Resale", "New Property", "Resale", "Resale", "New Property", "Resale", "Resale", "New Property"],
        "Ownership": ["Freehold", "Freehold", "Leasehold", "Freehold", "Freehold", "Freehold", "Leasehold", "Freehold"],
        "facing": ["North", "East", "South", "West", "North-East", "North", "East", "South-West"],
    })
    mock_data.to_csv(DATA_PATH, index=False)

print(f"Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = [str(col).strip() for col in df.columns]

# Support both spellings used by common copies of this dataset.
COLUMN_ALIASES = {
    "Amount(in rupees)": "Amount (in rupees)",
    "amount(in rupees)": "Amount (in rupees)",
    "Location": "location",
    "Facing": "facing",
}
df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})

required_columns = {
    "Amount (in rupees)", "Carpet Area", "Super Area", "Floor",
    "Bathroom", "Balcony", "location", "Furnishing",
    "Transaction", "Ownership", "facing",
}
missing = sorted(required_columns.difference(df.columns))
if missing:
    raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def parse_amount(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace("₹", "").replace("inr", "").replace(",", "")
    match = re.search(r"([\d.]+)", text)
    if not match:
        return np.nan
    amount = float(match.group(1))
    if "lac" in text or "lakh" in text:
        amount *= 1e5
    elif "cr" in text or "crore" in text:
        amount *= 1e7
    return amount


def parse_area(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower().strip().replace(",", "")
    match = re.search(r"([\d.]+)", text)
    if not match:
        return np.nan
    area = float(match.group(1))
    if "sqm" in text or "sq m" in text or "m²" in text:
        area *= 10.7639
    return area


def parse_floor(value):
    if pd.isna(value):
        return 1
    text = str(value).lower().strip()
    if "ground" in text or "basement" in text:
        return 0
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else 1


df["price_clean"] = df["Amount (in rupees)"].apply(parse_amount)
df["carpet_area_sqft"] = df["Carpet Area"].apply(parse_area)
df["super_area_sqft"] = df["Super Area"].apply(parse_area)
df["carpet_area_sqft"] = df["carpet_area_sqft"].fillna(df["super_area_sqft"])
df["floor_num"] = df["Floor"].apply(parse_floor)

for col in ["Bathroom", "Balcony"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["price_clean", "carpet_area_sqft"]).copy()
df["Bathroom"] = df["Bathroom"].fillna(1).clip(lower=1)
df["Balcony"] = df["Balcony"].fillna(0).clip(lower=0)

# Normalize empty categorical cells before grouping.
for col in ["location", "Furnishing", "Transaction", "Ownership", "facing"]:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    df.loc[df[col].eq(""), col] = "Unknown"

top_locations = df["location"].value_counts().nlargest(50).index.tolist()
df["location_grouped"] = df["location"].where(df["location"].isin(top_locations), "other")

numeric_features = ["carpet_area_sqft", "floor_num", "Bathroom", "Balcony"]
categorical_features = ["location_grouped", "Furnishing", "Transaction", "Ownership", "facing"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

X = df[numeric_features + categorical_features]
y = df["price_clean"]

pipeline = Pipeline([
    ("prep", preprocessor),
    ("reg", RandomForestRegressor(
        n_estimators=40,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )),
])

MAX_TRAIN_ROWS = 20_000
if len(df) > MAX_TRAIN_ROWS:
    sample_idx = df.sample(n=MAX_TRAIN_ROWS, random_state=42).index
    X = X.loc[sample_idx]
    y = y.loc[sample_idx]
    print(f"Dataset has {len(df):,} valid rows; using a reproducible sample of {MAX_TRAIN_ROWS:,} rows.")

print(f"Training on {len(X):,} rows...")
pipeline.fit(X, y)
joblib.dump(pipeline, MODEL_DIR / "house_price.pkl", compress=3)

allowed_locations = sorted(df["location_grouped"].unique().tolist())
for path in (BACKEND_LOCATIONS, FRONTEND_LOCATIONS):
    path.write_text(json.dumps(allowed_locations, indent=2), encoding="utf-8")

print(f"Model saved to: {MODEL_DIR / 'house_price.pkl'}")
print(f"Exported {len(allowed_locations)} location options.")
