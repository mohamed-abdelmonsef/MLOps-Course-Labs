# Bank Customer Churn Prediction API

## Overview

This project provides a FastAPI-based API to predict bank customer churn using a pre-trained machine learning model (Support Vector Classifier, SVC). The API includes endpoints for health checks and predictions, leveraging a dataset (`Churn_Modelling.csv`) that contains customer information and churn status. The project also includes MLflow integration for logging predictions and metrics.

The API implements three endpoints: a home endpoint (`/`), a health check (`/health`), and a prediction endpoint (`/predict`). It uses a pre-trained SVC model (`SVC_classifier.pkl`) and a transformer (`transformer.pkl`) for preprocessing input data.

## Features

- **FastAPI Application**: Provides a RESTful API with endpoints for prediction and health checks.
- **Data Preprocessing**: Uses a pre-trained transformer to encode categorical variables and scale numerical features.
- **Model Prediction**: Uses a pre-trained SVC model to predict customer churn.
- **MLflow Integration**: Logs prediction inputs and probabilities to MLflow for experiment tracking.
- **Testing**: Includes unit tests for all API endpoints.

## Installation

### Prerequisites

- Python 3.12+
- MLflow server running at `http://localhost:5000` (default tracking URI)
- The pre-trained model (`SVC_classifier.pkl`) and transformer (`transformer.pkl`) in `mlruns/models/`

### Steps

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd MLOPS-COURSE-LABS
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes packages like `fastapi`, `uvicorn`, `pandas`, `mlflow`, `joblib`, and `httpx`.

4. Ensure the MLflow server is running:

   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

## Usage

1. Place the `Churn_Modelling.csv` dataset in the `dataset/` directory (used for reference or retraining if needed).
2. Ensure the pre-trained model (`SVC_classifier.pkl`) and transformer (`transformer.pkl`) are in the `mlruns/models/` directory.
3. Run the FastAPI application:

   ```bash
   uvicorn app:app --reload
   ```
4. The API will be available at `http://localhost:8000`. You can access the following endpoints:
   - **Home (`/`)**: A welcome message.
   - **Health (`/health`)**: Check the API’s health status.
   - **Predict (`/predict`)**: Submit customer data to predict churn.
5. Use the interactive Swagger UI to test the API:
   - Open `http://localhost:8000/docs` in your browser.
   - Test the endpoints directly or use a tool like Postman.
6. View the experiment results in the MLflow UI:
   - Open `http://localhost:5000` in your browser.
   - Navigate to the `Churn_Prediction_exp` experiment to see logged predictions and metrics.

## File Structure

```
MLOPS-COURSE-LABS/
├── dataset/
│   └── Churn_Modelling.csv      # Input dataset (for reference)
├── mlruns/
│   └── models/
│       ├── SVC_classifier.pkl   # Pre-trained SVC model
│       └── transformer.pkl      # Pre-trained transformer
├── tests/
│   └── test_api.py              # Unit tests for API endpoints
├── app.py                       # FastAPI application
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies
```

## API Endpoints

1. **Home** (`GET /`):
   - Returns a welcome message and status.
   - Example Response: `{"message": "Welcome to Bank Churn Prediction API", "status": "running"}`
2. **Health** (`GET /health`):
   - Returns the health status of the API.
   - Example Response: `{"status": "healthy"}`
3. **Predict** (`POST /predict`):
   - Accepts raw customer data and returns a churn prediction.
   - Example Request Body:
     ```json
     {
       "CreditScore": 600,
       "Geography": "France",
       "Gender": "Male",
       "Age": 40,
       "Tenure": 3,
       "Balance": 50000,
       "NumOfProducts": 2,
       "HasCrCard": 1,
       "IsActiveMember": 1,
       "EstimatedSalary": 50000
     }
     ```
   - Example Response: `{"prediction": 0, "probability": 0.42, "message": "Prediction successful"}`

## Testing

1. Ensure the FastAPI app is running:
   ```bash
   uvicorn app:app --reload
   ```
2. Run the unit tests:
   ```bash
   pytest tests/ -v
   ```
3. Expected output should show 3 tests passing:
   ```
   tests/test_api.py::test_home_endpoint PASSED
   tests/test_api.py::test_health_endpoint PASSED
   tests/test_api.py::test_predict_endpoint PASSED
   ```

## Dependencies

The project relies on the following Python libraries (listed in `requirements.txt`):

- `fastapi`
- `uvicorn`
- `joblib`
- `pandas`
- `mlflow`
- `httpx`

## Notes

- Ensure the MLflow server is running before using the API to log predictions.
- The dataset `Churn_Modelling.csv` is not required for the API but can be used for retraining or reference.
- Adjust the `tracking_uri` in `app.py` if your MLflow server is hosted elsewhere.

