import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.metrics import root_mean_squared_error
import mlflow
import yaml

@pytest.fixture(autouse=True)
def mock_yaml_config():
    mock_config = {
        "mlflow": {
            "best_run_id": "test_run_id",
            "best_model_name": "test_model",
            "tracking_uri": "test_uri",
            "experiment_name": "test_experiment"
        }
    }
    with patch('yaml.safe_load', return_value=mock_config):
        yield mock_config

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [2, 4, 6]
    })
    y = pd.Series([2, 4, 6])
    return X, y

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([2, 4, 6])
    return model

@pytest.fixture(autouse=True)
def mock_mlflow():
    with patch('mlflow.xgboost.load_model') as mock_load:
        with patch('mlflow.get_run') as mock_get_run:
            mock_run = MagicMock()
            mock_run.data.params = {'param1': 'value1'}
            mock_get_run.return_value = mock_run
            mock_load.return_value = MagicMock()
            yield

def test_train_model(sample_data, mock_model):
    from src.models.train_model import train_model
    X, y = sample_data
    
    with patch('mlflow.xgboost.load_model', return_value=mock_model):
        model = train_model(X, y)
        
    assert model is not None
    mock_model.fit.assert_called_once()

def test_evaluate_model(sample_data, mock_model):
    from src.models.train_model import evaluate_model
    X, y = sample_data
    
    train_rmse, test_rmse = evaluate_model(X, mock_model, X, y, y)
    
    assert isinstance(train_rmse, float)
    assert isinstance(test_rmse, float)
    assert mock_model.predict.call_count == 2

@pytest.mark.parametrize("invalid_input,expected_error", [
    ((None, pd.Series([1, 2, 3])), TypeError),
    ((pd.DataFrame({'a': [1, 2, 3]}), None), TypeError),
    ((pd.DataFrame(), pd.Series([1, 2, 3])), ValueError),
    ((pd.DataFrame({'a': [1, 2, 3]}), pd.Series([1, 2])), ValueError),
])
def test_train_model_invalid_inputs(invalid_input, expected_error, mock_model):
    from src.models.train_model import train_model
    X, y = invalid_input
    
    with pytest.raises(expected_error):
        train_model(X, y)

def test_evaluate_model_with_different_predictions(sample_data):
    from src.models.train_model import evaluate_model
    X, y = sample_data
    
    model = MagicMock()
    model.predict.side_effect = [
        np.array([2.5, 4.5, 6.5]),  # Training predictions
        np.array([2.2, 4.2, 6.2])   # Testing predictions
    ]
    
    train_rmse, test_rmse = evaluate_model(X, model, X, y, y)
    
    assert train_rmse != test_rmse
    assert model.predict.call_count == 2