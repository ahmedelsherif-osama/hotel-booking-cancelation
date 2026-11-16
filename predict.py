from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "catboost_model.cbm"

FEATURE_COLUMNS = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "customer_type",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "customer_type",
]


# --------------------------------------------------
# Load model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train.py first.")

model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Hotel Booking Cancellation Prediction",
    description="Predict the probability that a hotel booking will be canceled.",
    version="1.0.0",
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------


class Booking(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    deposit_type: str
    agent: str
    customer_type: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "hotel": "City Hotel",
                "lead_time": 120,
                "arrival_date_year": 2017,
                "arrival_date_month": "July",
                "arrival_date_week_number": 27,
                "arrival_date_day_of_month": 5,
                "stays_in_weekend_nights": 2,
                "stays_in_week_nights": 3,
                "adults": 2,
                "children": 0,
                "babies": 0,
                "meal": "BB",
                "country": "PRT",
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
                "is_repeated_guest": 0,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0,
                "reserved_room_type": "A",
                "deposit_type": "No Deposit",
                "agent": "9",
                "customer_type": "Transient",
            }
        }
    }


# --------------------------------------------------
# Endpoints
# --------------------------------------------------


@app.get("/")
def home():
    return {"message": "Hotel Booking Cancellation Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(booking: Booking):

    # Convert input into a DataFrame
    booking_data = pd.DataFrame([booking.dict()])

    # Preserve training feature order
    booking_data = booking_data[FEATURE_COLUMNS]

    # Ensure categorical features are strings
    for col in CATEGORICAL_FEATURES:
        booking_data[col] = booking_data[col].astype(str)

    # Generate prediction
    probability = float(model.predict_proba(booking_data)[0, 1])

    prediction = int(probability >= 0.5)

    return {
        "cancellation_probability": round(probability, 4),
        "prediction": prediction,
        "prediction_label": ("Canceled" if prediction == 1 else "Not Canceled"),
    }
