from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from contextlib import asynccontextmanager
from pathlib import Path
import os
import pandas as pd
import mlflow
import yaml
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_config_path():
    """
    Get the configuration file path that works both locally and in Docker.
    
    Returns:
        Path: The path to the config file
    """
    # First try environment variable
    env_path = os.getenv('CONFIG_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
        
    # If env variable not set or file doesn't exist, try different relative paths
    possible_paths = [
        # When running from src/api directory
        Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml",
        # When running from project root
        Path("config") / "default.yaml",
        # When running in Docker
        Path("/app/config/default.yaml")
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # If no config file found, raise an error with helpful message
    raise FileNotFoundError(
        "Could not find config/default.yaml. Tried the following locations:\n" +
        "\n".join(f"- {p}" for p in possible_paths)
    )

# Load config using the helper function
try:
    config_path = get_config_path()
    logger.info(f"Loading config from: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError as e:
    logger.error(f"Configuration file not found: {str(e)}")
    raise
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}")
    raise

# Define feature names
FEATURE_NAMES = ['weather_code','temperature_2m_max','temperature_2m_min','temperature_2m_mean','apparent_temperature_max',
 'apparent_temperature_min','apparent_temperature_mean','daylight_duration','sunshine_duration','precipitation_sum',
 'rain_sum','precipitation_hours','wind_speed_10m_max','wind_gusts_10m_max','wind_direction_10m_dominant',
 'shortwave_radiation_sum','et0_fao_evapotranspiration','city','latitude','longitude','carbon_monoxide',
 'nitrogen_dioxide','sulphur_dioxide','ozone','aerosol_optical_depth','dust','uv_index','uv_index_clear_sky']

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
    description="API for predicting air quality metrics using weather and environmental data",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionInput(BaseModel):
    features: Dict[str, List[float]] = Field(
        ...,
        description="Dictionary containing feature arrays. Each feature must be provided as a list of float values.",
        example={
            "weather_code": [1, 3],
            "temperature_2m_max": [25.5, 26.7],
            "temperature_2m_min": [18.3,19.2],
            "temperature_2m_mean": [21.9, 22.5],
            "apparent_temperature_max": [27.0, 28.1],
            "apparent_temperature_min": [20.1, 20.8],
            "apparent_temperature_mean": [23.5, 24.0],
            "daylight_duration": [12.0,12.2],
            "sunshine_duration": [10.5, 10.7],
            "precipitation_sum": [5.3, 4.1],
            "rain_sum": [4.5,3.9],
            "precipitation_hours": [2,3],
            "wind_speed_10m_max": [15.0, 16.2],
            "wind_gusts_10m_max": [25.0,26.3],
            "wind_direction_10m_dominant": [180,190],
            "shortwave_radiation_sum": [200.5,105.0],
            "et0_fao_evapotranspiration": [3.1, 3.2],
            "city": [0,1],
            "latitude": [4.0483, 3.848],
            "longitude": [9.7043,11.5021],
            "carbon_monoxide": [0.1,0.12],
            "nitrogen_dioxide": [0.05, 0.06],
            "sulphur_dioxide": [0.02, 0.03],
            "ozone": [0.03,0.04],
            "aerosol_optical_depth": [0.15,0.16],
            "dust": [0.1,0.12],
            "uv_index": [8, 9],
            "uv_index_clear_sky": [10,11]
            }
    )

    model_config = ConfigDict(protected_namespaces=())

    def validate_features(self):
        """Validate that all required features are present and in the correct format"""
        if not all(feature in self.features for feature in FEATURE_NAMES):
            missing_features = set(FEATURE_NAMES) - set(self.features.keys())
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Validate that all arrays have the same length
        array_lengths = [len(arr) for arr in self.features.values()]
        if len(set(array_lengths)) > 1:
            raise ValueError("All feature arrays must have the same length")

class PredictionResponse(BaseModel):
    predictions: List[float] = Field(..., description="List of predicted air quality values")
    version: str = Field(..., description="Model version identifier")
    
    model_config = ConfigDict(protected_namespaces=())

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Air Quality Prediction API", 
        "version": app.state.model_version,
        "features": FEATURE_NAMES
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Make predictions for air quality based on provided features.
    
    Args:
        input_data (PredictionInput): Dictionary containing all required features as arrays
            All feature arrays must have the same length.
            
    Returns:
        PredictionResponse: Object containing predictions and model version
        
    Raises:
        HTTPException: If there's an error in input validation or prediction
    """
    try:
        # Validate input features
        input_data.validate_features()
        
        # Convert input dictionary to DataFrame with correct column order
        df = pd.DataFrame(input_data.features)[FEATURE_NAMES]
        
        # Make predictions
        predictions = app.state.model.predict(df)
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            version=app.state.model_version
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint returning service status and model version"""
    return {
        "status": "healthy",
        "version": app.state.model_version,
        "features_count": len(FEATURE_NAMES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )