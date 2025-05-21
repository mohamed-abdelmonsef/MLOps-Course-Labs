from fastapi.testclient import TestClient
from app import app
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = TestClient(app)

def test_home_endpoint():
    logger.info("Testing home endpoint")
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to Bank Churn Prediction API"
    assert response.json()["status"] == "running"

def test_health_endpoint():
    logger.info("Testing health endpoint")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    logger.info("Testing predict endpoint")
    sample_data = {
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
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert isinstance(response.json()["prediction"], int)
    assert isinstance(response.json()["probability"], float)