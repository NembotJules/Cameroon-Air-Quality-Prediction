from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from contextlib import asynccontextmanager
import pandas as pd
import mlflow
import yaml
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("../../config/default.yaml", "r") as file:
    config = yaml.safe_load(file)

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
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
    yield
    # Shutdown
    
app = FastAPI(
    title="Cameroon Air Quality Prediction API",
    description="API for predicting air quality metrics",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionInput(BaseModel):
    features: Dict[str, List[float]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    version: str  # Changed from model_version to version
    
    model_config = ConfigDict(protected_namespaces=())  # Disables protected namespace warning

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {"message": "Air Quality Prediction API", 
            "version": app.state.model_version}  # Updated to match the response model

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """_summary_

    Args:
        input_data (PredictionInput): Input features in the format expected by send_to_model_api

    Raises:
        HTTPException: _description_

    Returns:
        Predictions and model version
    """
    try:
        # Convert input dictionary to DataFrame
        df = pd.DataFrame(input_data.features)
        
        # Make predictions
        predictions = app.state.model.predict(df)
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            version=app.state.model_version  # Updated to match the field name
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": app.state.model_version}  # Updated for consistency



if __name__== "__main__": 
    import uvicorn
    uvicorn.run("app:app",
                 host="0.0.0.0",
                   port = 8000, 
                   reload = True)