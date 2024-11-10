import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_api_health():
    """Test the API health endpoint"""
    response = requests.get("http://localhost:8000/health")
    print("Health Check Response:", response.json())

def test_api_prediction():
    """Test the API prediction endpoint with sample data"""
    # Create sample data with the specified features
    sample_data = pd.DataFrame({
        # Weather features
        'weather_code': np.random.randint(0, 100, 3),
        'temperature_2m_max': np.random.uniform(20, 35, 3),
        'temperature_2m_min': np.random.uniform(15, 25, 3),
        'temperature_2m_mean': np.random.uniform(18, 30, 3),
        'apparent_temperature_max': np.random.uniform(22, 37, 3),
        'apparent_temperature_min': np.random.uniform(13, 23, 3),
        'apparent_temperature_mean': np.random.uniform(17, 30, 3),
        
        # Time features
        'daylight_duration': np.random.uniform(10*3600, 14*3600, 3),  # in seconds
        'sunshine_duration': np.random.uniform(7*3600, 12*3600, 3),  # in seconds
        
        # Precipitation features
        'precipitation_sum': np.random.uniform(0, 50, 3),
        'rain_sum': np.random.uniform(0, 45, 3),
        'precipitation_hours': np.random.uniform(0, 24, 3),
        
        # Wind features
        'wind_speed_10m_max': np.random.uniform(0, 20, 3),
        'wind_gusts_10m_max': np.random.uniform(5, 25, 3),
        'wind_direction_10m_dominant': np.random.uniform(0, 360, 3),
        
        # Radiation and evaporation
        'shortwave_radiation_sum': np.random.uniform(0, 1000, 3),
        'et0_fao_evapotranspiration': np.random.uniform(0, 10, 3),
        
        # Location features
        'city': np.random.choice(['Yaounde', 'Douala', 'Bamenda'], 3),
        'latitude': np.random.uniform(2, 13, 3),
        'longitude': np.random.uniform(8, 17, 3),
        
        # Air quality parameters
        'carbon_monoxide': np.random.uniform(0, 10, 3),
        'nitrogen_dioxide': np.random.uniform(0, 200, 3),
        'sulphur_dioxide': np.random.uniform(0, 100, 3),
        'ozone': np.random.uniform(0, 100, 3),
        'aerosol_optical_depth': np.random.uniform(0, 1, 3),
        'dust': np.random.uniform(0, 100, 3),
        
        # UV indices
        'uv_index': np.random.uniform(0, 11, 3),
        'uv_index_clear_sky': np.random.uniform(0, 12, 3)
    })
    
    # Convert DataFrame to the format expected by the API
    features_list = sample_data.to_dict(orient='records')
    
    # Prepare the request payload
    payload = {
        'features': features_list
    }
    
    # Send POST request to the API
    try:
        response = requests.post(
            "http://0.0.0.0:8000/predict",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        
        # Print the response
        print("\nPrediction Response:", response.json())
        
        # Print shape of the data sent
        print(f"\nData shape: {sample_data.shape}")
        print(f"Number of features: {len(sample_data.columns)}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {str(e)}")

def print_feature_ranges():
    """Print the ranges of values used for each feature"""
    print("\nFeature Ranges Used:")
    print("-" * 50)
    print("\nTemperature features:")
    print("Temperature 2m Max: 20-35°C")
    print("Temperature 2m Min: 15-25°C")
    print("Temperature 2m Mean: 18-30°C")
    print("Apparent Temperature Max: 22-37°C")
    print("Apparent Temperature Min: 13-23°C")
    print("Apparent Temperature Mean: 17-30°C")
    
    print("\nPrecipitation features:")
    print("Precipitation Sum: 0-50 mm")
    print("Rain Sum: 0-45 mm")
    
    print("Precipitation Hours: 0-24 hours")
    
    print("\nWind features:")
    print("Wind Speed 10m Max: 0-20 m/s")
    print("Wind Gusts 10m Max: 5-25 m/s")
    print("Wind Direction 10m Dominant: 0-360 degrees")
    
    print("\nAir quality parameters:")
    print("Carbon Monoxide: 0-10 ppm")
    print("Nitrogen Dioxide: 0-200 ppb")
    print("Sulphur Dioxide: 0-100 ppb")
    print("Ozone: 0-100 ppb")
    print("Aerosol Optical Depth: 0-1")
    print("Dust: 0-100 µg/m³")
    
    print("\nLocation features:")
    print("Latitude: 2-13 (Cameroon)")
    print("Longitude: 8-17 (Cameroon)")
    print("Cities: Yaounde, Douala, Bamenda")

if __name__ == "__main__":
    # Test health endpoint
    test_api_health()
    
    # Print feature ranges
    print_feature_ranges()
    
    # Test prediction endpoint
    test_api_prediction()