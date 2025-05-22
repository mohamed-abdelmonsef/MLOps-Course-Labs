import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Bank Churn Prediction API")
# Middleware for logging requests
Instrumentator().instrument(app).expose(app)

# Load the trained model and transformer
BASE_DIR = Path(__file__).resolve().parent / "models"
model_path = BASE_DIR / "SVC_classifier.pkl"
transformer_path = BASE_DIR / "transformer.pkl"

# Load the trained model and transformer
logger.info(f"Loading model from {model_path} and transformer from {transformer_path}")
if not model_path.exists() or not transformer_path.exists():
    raise FileNotFoundError("Model or transformer file not found")
model = joblib.load(model_path)
transformer = joblib.load(transformer_path)

# MLflow client for logging
client = MlflowClient()
experiment_name = "Churn_Prediction_exp"
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = client.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

# Home endpoint
@app.get("/")
async def home():
    logger.info("Accessed home endpoint")
    return {"message": "Welcome to Bank Churn Prediction API", "status": "running"}

# Health endpoint
@app.get("/health")
async def health():
    logger.info("Accessed health endpoint")
    return {"status": "healthy"}


# Pydantic model for the predict endpoint input
class PredictionInput(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

    class Config:
        schema_extra = {
            "example": {
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
        }

# Predict endpoint with Pydantic model
@app.post("/predict")
async def predict(data: PredictionInput):
    logger.info(f"Received prediction request with data: {data.dict()}")
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data.model_dump()])
        required_cols = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", 
                        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required columns")

        # Preprocess data using the loaded transformer
        X = df.drop("Exited", axis=1, errors='ignore')
        X_transformed = transformer.transform(X)
        X_transformed_df = pd.DataFrame(X_transformed, columns=transformer.get_feature_names_out())

        # Make prediction
        prediction = model.predict(X_transformed_df)
        probability = model.predict_proba(X_transformed_df)[:, 1][0]
        logger.info(f"Prediction made: {prediction[0]}, Probability: {probability}")

        # Log to MLflow
        with mlflow.start_run(experiment_id=experiment_id, run_name="API_Prediction"):
            mlflow.log_param("input_data", str(data.dict()))
            mlflow.log_metric("prediction_probability", float(probability))

        return JSONResponse({
            "prediction": int(prediction[0]),
            "probability": float(probability),
            "message": "Prediction successful"
        })
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))