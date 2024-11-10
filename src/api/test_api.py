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
    sample_data = {
    "weather_code": [
      1,
      2
    ],
    "temperature_2m_max": [
      25.5,
      26.7
    ],
    "temperature_2m_min": [
      18.3,
      19.1
    ],
    "temperature_2m_mean": [
      21.9,
      22.5
    ],
    "apparent_temperature_max": [
      27.0,
      28.1
    ],
    "apparent_temperature_min": [
      20.1,
      20.8
    ],
    "apparent_temperature_mean": [
      23.5,
      24.0
    ],
    "daylight_duration": [
      12.0,
      12.2
    ],
    "sunshine_duration": [
      10.5,
      10.7
    ],
    "precipitation_sum": [
      5.3,
      4.1
    ],
    "rain_sum": [
      4.5,
      3.9
    ],
    "precipitation_hours": [
      2,
      3
    ],
    "wind_speed_10m_max": [
      15.0,
      16.2
    ],
    "wind_gusts_10m_max": [
      25.0,
      26.5
    ],
    "wind_direction_10m_dominant": [
      180,
      190
    ],
    "shortwave_radiation_sum": [
      200.5,
      205.0
    ],
    "et0_fao_evapotranspiration": [
      3.1,
      3.2
    ],
    "city": [
      0,
      1
    ],
    "latitude": [
      4.0483,
      3.848
    ],
    "longitude": [
      9.7043,
      11.5021
    ],
    "carbon_monoxide": [
      0.1,
      0.12
    ],
    "nitrogen_dioxide": [
      0.05,
      0.06
    ],
    "sulphur_dioxide": [
      0.02,
      0.03
    ],
    "ozone": [
      0.03,
      0.04
    ],
    "aerosol_optical_depth": [
      0.15,
      0.16
    ],
    "dust": [
      0.1,
      0.12
    ],
    "uv_index": [
      8,
      9
    ],
    "uv_index_clear_sky": [
      10,
      11
    ]
  }
    
    # Convert DataFrame to the format expected by the API
    #features_list = sample_data.to_dict(orient='records')
    
    # Prepare the request payload
    payload = {
        'features': sample_data
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