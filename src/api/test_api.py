import requests
import pandas as pd
import numpy as np

def test_api_health():
    """Test the API health endpoint"""
    response = requests.get("http://localhost:8000/health")
    print("Health Check Response:", response.json())

def test_api_prediction():
    """Test the API prediction endpoint with sample data"""
    # Create sample data 
    sample_data = pd.DataFrame({
        'feature1': np.random.rand(3),
        'feature2': np.random.rand(3),
        'feature3': np.random.rand(3)
    })
    
    # Converting DataFrame to the format expected by the API
    features_list = sample_data.to_dict(orient='records')
    
    # Prepare the request payload
    payload = {
        'features': features_list
    }
    
    # Send POST request to the API
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        
        # Print the response
        print("\nPrediction Response:", response.json())
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {str(e)}")

if __name__ == "__main__":
    # Test health endpoint
    test_api_health()
    
    # Test prediction endpoint
    test_api_prediction()