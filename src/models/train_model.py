import mlflow
import pandas as pd
import yaml
from sklearn.metrics import root_mean_squared_error
import os
import xgboost as xgb
from typing import Union, Tuple


def update_config_file(config_path: str, new_run_id: str, test_rmse: float, 
                      current_best_rmse: float = float('inf')) -> None:
    """
    Update the config file with a new run_id if the new model performs better.
    
    Args:
        config_path (str): Path to the config file
        new_run_id (str): MLflow run ID of the new model
        test_rmse (float): RMSE of the new model
        current_best_rmse (float): RMSE of the current best model
    """
    print("\nModel Comparison:")
    print(f"Current best RMSE: {current_best_rmse}")
    print(f"New model RMSE: {test_rmse}")
    
    # Check if the new model is better
    if current_best_rmse == float('inf'):
        print("No previous best model found or error retrieving previous RMSE.")
        should_update = True
    else:
        improvement = abs(current_best_rmse) - abs(test_rmse)
        print(f"Improvement in RMSE: {improvement}")
        #because test_rmse and current_best_rmse are both negative values...
        should_update = test_rmse > current_best_rmse
    
    if should_update:
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Update the best_run_id
            old_run_id = config['mlflow']['best_run_id']
            config['mlflow']['best_run_id'] = new_run_id
            
            # Save the updated config
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
                
            print(f"\nConfig updated successfully:")
            print(f"Old run ID: {old_run_id}")
            print(f"New run ID: {new_run_id}")
            print(f"New best RMSE: {test_rmse}")
            
        except Exception as e:
            print(f"Error updating config file: {str(e)}")
    else:
        print("\nNo update needed - current model did not improve upon best model")


# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the config file
default_config_name = os.path.join(current_dir, '..', '..', 'config', 'default.yaml')

with open(default_config_name, "r") as file: 
    default_config = yaml.safe_load(file)

best_run_id = default_config["mlflow"]["best_run_id"]
best_model_name = default_config["mlflow"]["best_model_name"]
mlflow.set_tracking_uri(default_config["mlflow"]["tracking_uri"])
mlflow.set_experiment(default_config["mlflow"]["experiment_name"])


def train_model(X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> xgb.XGBRegressor: 
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


def evaluate_model(X: pd.DataFrame, model: xgb.XGBRegressor, 
                  test_data: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], 
                  test_target: Union[pd.Series, pd.DataFrame]) -> Tuple[float, float]: 
    # Evaluate training performance 
    y_train_pred = model.predict(X)
    train_rmse = root_mean_squared_error(y, y_train_pred)
    print(f'Training RMSE: {train_rmse}')
    
    # Evaluate testing performance 
    y_test_pred = model.predict(test_data)
    test_rmse = root_mean_squared_error(test_target, y_test_pred)
    print(f'Testing RMSE: {test_rmse}')
  
    return train_rmse, test_rmse

def get_current_best_rmse() -> float:
    """
    Get the RMSE of the current best model from MLflow.
    
    Returns:
        float: RMSE of the current best model, or inf if no previous model exists
    """
    try:
        # Get the run
        run = mlflow.get_run(best_run_id)
        
        # Debug prints
        print(f"Retrieved run with ID: {best_run_id}")
        print("Available metrics:", run.data.metrics)
        
        # Check if metrics exist
        if not run.data.metrics:
            print("No metrics found in the run")
            return float('inf')
            
        # Try to get the test_rmse
        test_rmse = run.data.metrics.get('test_rmse')
        
        if test_rmse is None:
            print("test_rmse not found in metrics")
            return float('inf')
            
        print(f"Found test_rmse: {test_rmse}")
        return float(test_rmse)
        
    except Exception as e:
        print(f"Error retrieving best RMSE: {str(e)}")
        return float('inf')

if __name__ == "__main__": 
    with mlflow.start_run() as run:
        # Get the current run ID
        current_run_id = run.info.run_id
        
        # Load and prepare data
        X = pd.read_csv(default_config["data"]["preprocessed_train_data_path"])
        y = pd.read_csv(default_config["data"]["preprocessed_train_target_path"])
        test_data = pd.read_csv(default_config["data"]["preprocessed_test_data_path"])
        test_y = pd.read_csv(default_config["data"]["preprocessed_test_target_path"])
        
        # Train and evaluate model
        model = train_model(X, y)
        train_rmse, test_rmse = evaluate_model(X, model, test_data, y, test_y)  

        # Log metrics
        print("\nLogging metrics to MLflow:")
        print(f"Train RMSE: {train_rmse}")
        print(f"Test RMSE: {test_rmse}")
        mlflow.log_metric('train_rmse', train_rmse)
        mlflow.log_metric('test_rmse', test_rmse)

        # Log model
        print("\nLogging model to MLflow...")
        mlflow.xgboost.log_model(
            registered_model_name=best_model_name,
            artifact_path=best_model_name,
            xgb_model=model,
            input_example=X
        )

        # Get current best RMSE and update config if new model is better
        print("\nChecking current best model performance...")
        current_best_rmse = get_current_best_rmse()
        update_config_file(default_config_name, current_run_id, test_rmse, current_best_rmse)