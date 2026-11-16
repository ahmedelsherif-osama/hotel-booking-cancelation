
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hotel_bookings.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "catboost_model.cbm"


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    return df


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_data(df):
    # Exclude no-show reservations
    df = df[df["reservation_status"] != "No-Show"].copy()

    # Remove rows with missing children values
    df = df.dropna(subset=["children"]).copy()

    # Drop excluded and unavailable-at-booking-time features
    columns_to_drop = [
        "company",
        "reservation_status",
        "reservation_status_date",
        "assigned_room_type",
        "booking_changes",
        "days_in_waiting_list",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "adr"
    ]

    df = df.drop(columns=columns_to_drop)

    # Handle missing categorical values
    df["country"] = (
        df["country"]
        .fillna("Unknown")
        .astype(str)
    )

    df["agent"] = (
        pd.to_numeric(df["agent"], errors="coerce")
        .astype("Int64")
        .astype("string")
        .fillna("Unknown")
    )

    # Separate features and target
    target = "is_canceled"

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Identify categorical features
    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # Convert categorical features to strings
    for col in categorical_features:
        X[col] = X[col].fillna("Unknown").astype(str)

    print("Processed dataset shape:", X.shape)
    print("Number of features:", len(X.columns))
    print("Categorical features:", categorical_features)

    return X, y, categorical_features


# --------------------------------------------------
# Train model
# --------------------------------------------------

def train_model(X, y, categorical_features):
    model = CatBoostClassifier(
        iterations=300,
        depth=9,
        learning_rate=0.05,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False
    )

    model.fit(
        X,
        y,
        cat_features=categorical_features
    )

    return model


# --------------------------------------------------
# Save model
# --------------------------------------------------

def save_model(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_PATH))

    print(f"Model saved to: {MODEL_PATH}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    df = load_data()

    X, y, categorical_features = preprocess_data(df)

    model = train_model(
        X,
        y,
        categorical_features
    )

    save_model(model)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
