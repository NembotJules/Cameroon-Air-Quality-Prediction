from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import yaml
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("config/default.yaml", "r") as file:
    config = yaml.safe_load(file)

app = FastAPI(
    title="Cameroon Air Quality Prediction API",
    description="API for predicting air quality metrics",
    version="1.0.0"
)

class PredictionInput(BaseModel):
    features: Dict[str, List[float]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str

# Load model at startup
@app.on_event("startup")
async def load_model():
    try:
        logger.info("Loading model...")
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        logged_model = f'runs:/{config["mlflow"]["best_run_id"]}/{config["mlflow"]["best_model_name"]}'
        app.state.model = mlflow.xgboost.load_model(logged_model)
        app.state.model_version = config["mlflow"]["best_run_id"]
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Air Quality Prediction API", 
            "model_version": app.state.model_version}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    try:
        # Convert input dictionary to DataFrame
        df = pd.DataFrame(input_data.features)
        
        # Make predictions
        predictions = app.state.model.predict(df)
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            model_version=app.state.model_version
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": app.state.model_version}