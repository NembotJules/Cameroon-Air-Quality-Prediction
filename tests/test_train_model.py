import os
import pytest
import yaml
import pandas as pd
import mlflow
import xgboost as xgb
import tempfile
import shutil

# Import the functions to be tested
from src.models.train_model import (
    update_config_file,
    train_model,
    evaluate_model,
    get_current_best_rmse
)

@pytest.fixture
def sample_config():
    """Create a temporary config file for testing."""
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, 'test_config.yaml')
    
    # Create a sample config
    config_data = {
        'mlflow': {
            'best_run_id': 'initial_run_id',
            'best_model_name': 'test_model',
            'tracking_uri': 'test_uri',
            'experiment_name': 'test_experiment'
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    yield config_path
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_update_config_file(sample_config):
    """
    Test the update_config_file function with different scenarios.
    """
    # Scenario 1: First model (no previous best model)
    update_config_file(sample_config, 'new_run_id', -0.85)
    
    with open(sample_config, 'r') as f:
        updated_config = yaml.safe_load(f)
    
    assert updated_config['mlflow']['best_run_id'] == 'new_run_id'
    
    # Scenario 2: Better model (lower RMSE)
    update_config_file(sample_config, 'better_run_id', -0.90, current_best_rmse=-0.85)
    
    with open(sample_config, 'r') as f:
        updated_config = yaml.safe_load(f)
    
    assert updated_config['mlflow']['best_run_id'] == 'better_run_id'
    
    # Scenario 3: Worse model (higher RMSE)
    original_run_id = updated_config['mlflow']['best_run_id']
    update_config_file(sample_config, 'worse_run_id', -0.80, current_best_rmse=-0.90)
    
    with open(sample_config, 'r') as f:
        updated_config = yaml.safe_load(f)
    
    assert updated_config['mlflow']['best_run_id'] == original_run_id

def test_train_model_input_validation():
    """
    Test input validation for train_model function.
    """
    # Prepare sample data
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([10, 20, 30])
    
    # Test successful training
    model = train_model(X, y)
    assert isinstance(model, xgb.XGBRegressor)
    
    # Test input type errors
    with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
        train_model(list(X.values), y)
    
    with pytest.raises(TypeError, match="y must be a pandas Series or DataFrame"):
        train_model(X, list(y.values))
    
    # Test empty input errors
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        train_model(pd.DataFrame(), pd.Series())
    
    # Test mismatched length
    with pytest.raises(ValueError, match="X and y must have the same number of samples"):
        train_model(X, pd.Series([10, 20]))

def test_evaluate_model():
    """
    Test the evaluate_model function.
    """
    # Prepare sample data
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([10, 20, 30])
    test_data = pd.DataFrame({'feature1': [4, 5], 'feature2': [7, 8]})
    test_target = pd.Series([40, 50])
    
    # Train a sample model
    model = train_model(X, y)
    
    # Evaluate the model
    train_rmse, test_rmse = evaluate_model(X, model, test_data, y, test_target)
    
    # Check RMSE values (just basic checks)
    assert isinstance(train_rmse, float)
    assert isinstance(test_rmse, float)
    assert train_rmse >= 0
    assert test_rmse >= 0

def test_get_current_best_rmse(mocker):
    """
    Test the get_current_best_rmse function.
    
    Note: This test uses mocker to simulate MLflow behavior.
    Requires pytest-mock to be installed.
    """
    # Mock the mlflow.get_run to return a predefined run
    mock_run = mocker.Mock()
    mock_run.data.metrics = {'test_rmse': 0.85}
    
    mocker.patch('mlflow.get_run', return_value=mock_run)
    
    # Test retrieval of RMSE
    rmse = get_current_best_rmse()
    assert rmse == 0.85
    
    # Test scenario with no metrics
    mock_run.data.metrics = {}
    rmse = get_current_best_rmse()
    assert rmse == float('inf')
    
    # Test exception handling
    mocker.patch('mlflow.get_run', side_effect=Exception("MLflow error"))
    rmse = get_current_best_rmse()
    assert rmse == float('inf')

# Optional: Add more comprehensive tests or edge cases as needed