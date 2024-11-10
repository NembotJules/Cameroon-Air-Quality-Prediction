from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
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
            "temperature_2m_max": [25.5, 26.7],
            "temperature_2m_min": [18.3, 19.1],
            # Add more examples as needed
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