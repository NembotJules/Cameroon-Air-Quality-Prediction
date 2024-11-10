import mlflow
import pandas as pd
import yaml
from sklearn.metrics import root_mean_squared_error
import os
import xgboost as xgb
from typing import Union, Tuple


# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the config file
default_config_name = os.path.join(current_dir, '..', '..', 'config', 'default.yaml')

#default_config_name = "../../config/default.yaml"

with open(default_config_name, "r") as file: 
    default_config = yaml.safe_load(file)


best_run_id = default_config["mlflow"]["best_run_id"]
best_model_name = default_config["mlflow"]["best_model_name"]
mlflow.set_tracking_uri(default_config["mlflow"]["tracking_uri"])
mlflow.set_experiment(default_config["mlflow"]["experiment_name"])



def train_model(X:pd.DataFrame, y:Union[pd.Series, pd.DataFrame]) -> xgb.XGBRegressor: 
    """
    Train the model with input validation.

    This function will be the one to trigger for automatic model retraining
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series or pd.DataFrame): Training target
        
    Returns:
        xgb.XGBRegressor: Trained XGBoost model
        
    Raises:
        TypeError: If inputs are not pandas DataFrame/Series
        ValueError: If inputs are empty or have different lengths
    """

    # Input validation
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series or DataFrame")
        
    if X.empty or y.empty:
        raise ValueError("Input data cannot be empty")
        
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    logged_model = f'runs:/{best_run_id}/{best_model_name}'
# Load model as a PyFuncModel.
    model = mlflow.xgboost.load_model(logged_model)

     # Retrieve the run that created this model
    run = mlflow.get_run(best_run_id)
    
    # Print the parameters used during training
    print("Model parameters:")
    for param, value in run.data.params.items():
        print(f"{param}: {value}")

    model.fit(X, y)

    return model


def evaluate_model(X:pd.DataFrame, model:xgb.XGBRegressor, 
                   test_data:pd.DataFrame, y:Union[pd.Series, pd.DataFrame], 
                   test_target:Union[pd.Series, pd.DataFrame]) -> Tuple[float, float]: 
    # Evaluate training performance 
    y_train_pred = model.predict(X)
    train_rmse = root_mean_squared_error(y, y_train_pred)
    print(f'Training RMSE: {train_rmse}')
    # Evaluate testing performance 
    y_test_pred = model.predict(test_data)
    test_rmse = root_mean_squared_error(test_target, y_test_pred)
    print(f'Testing RMSE: {test_rmse}')
  
    return train_rmse, test_rmse



if __name__ == "__main__": 

    with mlflow.start_run():
        X = pd.read_csv(default_config["data"]["preprocessed_train_data_path"])
        y = pd.read_csv(default_config["data"]["preprocessed_train_target_path"])
        test_data = pd.read_csv(default_config["data"]["preprocessed_test_data_path"])
        test_target = pd.read_csv(default_config["data"]["preprocessed_test_target_path"])
        model = train_model(X, y)
        train_rmse, test_rmse =  evaluate_model(X, model, test_data, y, test_target)  

        mlflow.log_metric('train_rmse', train_rmse)
        mlflow.log_metric('test_rmse', test_rmse)

        mlflow.xgboost.log_model(
                registered_model_name = best_model_name,
                artifact_path = best_model_name,
                xgb_model = model,
                input_example = X)