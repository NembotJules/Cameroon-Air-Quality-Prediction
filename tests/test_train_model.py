import pytest
import pandas as pd
import yaml
import os
from unittest.mock import MagicMock, patch
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from src.models.train_model import (
    train_model,
    evaluate_model,
    update_config_file,
    get_current_best_rmse,
)

# Fixtures for test data and config
@pytest.fixture
def dummy_data():
    # Ensure 28 features for the model
    X = pd.DataFrame({f"feature{i}": range(1, 4) for i in range(1, 29)})
    y = pd.Series([7, 8, 9])
    return X, y

@pytest.fixture
def dummy_test_data():
    test_X = pd.DataFrame({f"feature{i}": range(10, 12) for i in range(1, 29)})
    test_y = pd.Series([14, 15])
    return test_X, test_y

# @pytest.fixture
# def dummy_config(tmp_path):
#     config_path = tmp_path / "default.yaml"
#     config_data = {
#         "mlflow": {
#             "best_run_id": "test_run_id",
#             "best_model_name": "test_model",
#             "tracking_uri": "http://test_tracking_uri",
#             "experiment_name": "test_experiment",
#         },
#     }
#     with open(config_path, "w") as file:
#         yaml.dump(config_data, file)
#     return config_path, config_data



def test_train_model_invalid_input():
    with pytest.raises(TypeError):
        train_model([1, 2, 3], pd.Series([4, 5, 6]))  # Invalid X type

    with pytest.raises(TypeError):
        train_model(pd.DataFrame({"a": [1, 2]}), [4, 5])  # Invalid y type

    with pytest.raises(ValueError):
        train_model(pd.DataFrame(), pd.Series())  # Empty data

    with pytest.raises(ValueError):
        train_model(pd.DataFrame({"a": [1]}), pd.Series([2, 3]))  # Mismatched lengths

# Test evaluate_model function
def test_evaluate_model(dummy_data, dummy_test_data):
    X, y = dummy_data
    test_X, test_y = dummy_test_data

    mock_model = MagicMock(spec=XGBRegressor)
    mock_model.predict.side_effect = [
        y.values,  # Mock predictions for training
        test_y.values,  # Mock predictions for testing
    ]

    train_rmse, test_rmse = evaluate_model(X, mock_model, test_X, y, test_y)

    assert train_rmse == pytest.approx(mean_squared_error(y, y, squared=False))
    assert test_rmse == pytest.approx(mean_squared_error(test_y, test_y, squared=False))

#
# Test get_current_best_rmse function
def test_get_current_best_rmse():
    mock_run = MagicMock()
    mock_run.data.metrics = {"test_rmse": 1.5}

    with patch("mlflow.get_run", return_value=mock_run):
        rmse = get_current_best_rmse()
        assert rmse == 1.5

def test_get_current_best_rmse_no_metrics():
    mock_run = MagicMock()
    mock_run.data.metrics = {}

    with patch("mlflow.get_run", return_value=mock_run):
        rmse = get_current_best_rmse()
        assert rmse == float("inf")
