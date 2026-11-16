# Hotel Booking Cancellation Prediction

## Problem Description

Hotel booking cancellations create uncertainty for hotel operators. They can lead to lost revenue, inefficient room allocation, and difficulties in managing occupancy.

This project develops a machine learning model that predicts the probability that a hotel booking will be canceled, using information available at the time of reservation.

The prediction can help hotels identify bookings with a higher cancellation risk and support reservation and revenue management decisions.

The project uses the **Hotel Booking Demand** dataset, which contains booking records for a city hotel and a resort hotel from 2015 to 2017.

The task is formulated as a binary classification problem:

- `0`: Booking not canceled.
- `1`: Booking canceled.

No-show reservations are excluded from the analysis.

## Dataset

The project uses the [Hotel Booking Demand dataset on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

Download `hotel_bookings.csv` and place it inside the `data/` directory:

```text
data/
└── hotel_bookings.csv
```

The dataset contains reservation details, including booking lead time, arrival dates, length of stay, guest information, distribution channel, and deposit type.

## Exploratory Data Analysis and Data Preparation

The notebook covers:

- Dataset structure, data types, and descriptive statistics.
- Missing-value analysis and data cleaning.
- Target variable distribution and cancellation patterns.
- Analysis of numerical and categorical features.
- Feature importance analysis.
- Comparison of original and engineered feature sets.

Preprocessing includes handling missing categorical values, converting agent identifiers to categorical values, and removing columns that are unavailable at booking time or could introduce information leakage.

The target variable is `is_canceled`.

## Modeling Approach

Several classification algorithms were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- CatBoost

Both original and engineered feature sets were evaluated.

Model performance was compared using ROC-AUC, PR-AUC, precision, recall, and F1-score. Hyperparameter tuning was performed for Logistic Regression and CatBoost using cross-validation.

**CatBoost with the original feature set** was selected as the final model based on its validation performance.

### Final Model Performance

The final CatBoost model was evaluated on a held-out test set.

| Metric | Test Score |
|---|---:|
| ROC-AUC | 0.9197 |
| PR-AUC | 0.8735 |
| Precision | 0.7985 |
| Recall | 0.7185 |
| F1-score | 0.7564 |
| Accuracy | 0.84 |

The model achieves strong discrimination between canceled and non-canceled bookings on the held-out test set.

## Project Structure

```text
hotel_booking_demand/
├── data/
│   └── hotel_bookings.csv
├── models/
│   └── catboost_model.cbm
├── notebooks/
│   └── hotel_booking_analysis.ipynb
├── train.py
├── predict.py
├── requirements.txt
├── requirements-docker.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10 or newer
- pip
- Python `venv` module
- Docker (for containerized deployment)

Check your Python and pip installations:

```bash
python --version
python -m pip --version
```

If Python is not installed, download it from [python.org](https://www.python.org/downloads/).

## Installation

Clone the repository:

```bash
git clone https://github.com/ahmedelsherif-osama/hotel-booking-cancelation.git
cd hotel-booking-cancelation
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Linux / macOS:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Data Analysis

Start Jupyter Notebook:

```bash
jupyter notebook notebooks/hotel_booking_analysis.ipynb
```

Run the notebook cells to explore the dataset, preprocess the data, perform EDA, compare models, tune hyperparameters, and evaluate the final model.

### 2. Model Training

To reproduce the final model training process:

```bash
python train.py
```

The script preprocesses the dataset, trains the selected CatBoost model, and saves the trained model to:

```text
models/catboost_model.cbm
```

The dataset must be available in `data/hotel_bookings.csv`.

### 3. Prediction API

The API uses the saved CatBoost model to generate predictions.

Start the FastAPI service from the project root:

```bash
uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
```

The API will be available at:

- API root: http://127.0.0.1:9696/
- Health check: http://127.0.0.1:9696/health
- Interactive API documentation: http://127.0.0.1:9696/docs

### API Usage

The API exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict` | Predict booking cancellation |

Example prediction request:

```bash
curl -X POST "http://127.0.0.1:9696/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "customer_type": "Transient"
  }'
```

The response contains:

- `cancellation_probability`: estimated probability of cancellation.
- `prediction`: predicted class (`0` or `1`).
- `prediction_label`: human-readable prediction.

The default classification threshold is 0.5.

Alternatively, use the interactive Swagger documentation at `/docs` to submit a prediction request.

### 4. Docker

The Docker image packages the prediction API and saved model. It uses `requirements-docker.txt` to install only the dependencies needed at runtime.

Build the image:

```bash
docker build -t hotel-cancellation-prediction .
```

Run the container:

```bash
docker run --rm -p 9696:9696 hotel-cancellation-prediction
```

Access the API documentation at:

http://127.0.0.1:9696/docs

The container listens on the port provided by the `PORT` environment variable, defaulting to `9696`.

## Cloud Deployment

**Deployment platform:** Render

**Live API URL:** To be added after deployment.

The deployed service will expose the same prediction endpoints as the local Dockerized application.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- CatBoost
- XGBoost
- FastAPI
- Pydantic
- Uvicorn
- Jupyter Notebook
- Docker
- Render

## Limitations

- The dataset contains historical hotel booking records from 2015–2017; booking patterns may change over time.
- Predictions depend on the quality and availability of booking information supplied to the API.
- A predicted cancellation probability is an estimate, not a guarantee that a booking will or will not be canceled.

## Author

Ahmed Elsherif