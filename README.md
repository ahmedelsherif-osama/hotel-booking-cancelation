# Hotel Booking Cancellation Prediction

## Problem Description

This project aims to predict the probability that a hotel booking will be canceled based on information available at the time of reservation.

The goal is to help hotels identify bookings with a higher cancellation risk and support better reservation and revenue management decisions.

The project uses the Hotel Booking Demand dataset, containing booking records for a city hotel and a resort hotel from 2015 to 2017.

The task is formulated as a binary classification problem:

- `0`: Booking not canceled.
- `1`: Booking canceled.

No-show reservations are excluded from the analysis.

## Dataset

Download the dataset from [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

Place the CSV file in the project root directory.

**Note:** The dataset is also uploaded to the data folder.

## Project Structure

```text
.
├── data/
│   └── hotel_bookings.csv
├── notebooks/
│   └── hotel_booking_analysis.ipynb
├── train.py
├── predict.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ahmedelsherif-osama/hotel-booking-cancelation.git
cd hotel-booking-cancelation
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Data Analysis

Open the notebook:

```bash
jupyter notebook notebooks/hotel_booking_analysis.ipynb
```

Run the notebook cells to explore the dataset, preprocess the data, and evaluate candidate models.

### 2. Model Training

Train and save the selected model:

```bash
python train.py
```

### 3. Prediction API

Start the prediction service:

```bash
python predict.py
```

The API accepts booking information and returns a cancellation prediction and estimated cancellation probability.

### 4. Docker

Build the Docker image:

```bash
docker build -t hotel-cancellation-prediction .
```

Run the container:

```bash
docker run -p 9696:9696 hotel-cancellation-prediction
```

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- CatBoost / XGBoost
- Flask or FastAPI
- Docker
