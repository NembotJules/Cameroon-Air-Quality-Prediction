import mlflow
import pandas as pd
import yaml
from sklearn.metrics import root_mean_squared_error

default_config_name = "../../config/default.yaml"

with open(default_config_name, "r") as file: 
    default_config = yaml.safe_load(file)


best_run_id = default_config["mlflow"]["best_run_id"]
best_model_name = default_config["mlflow"]["best_model_name"]

mlflow.set_tracking_uri(default_config["mlflow"]["tracking_uri"])
mlflow.set_experiment(default_config["mlflow"]["experiment_name"])

def train_model(X, y): 

    logged_model = f'runs:/{best_run_id}/{best_model_name}'

    # Load model as a PyFuncModel.
    XGBR2 = mlflow.xgboost.load_model(logged_model)

    # Train the model on the full dataset
    XGBR2.fit(X, y) 

    return XGBR2


def evaluate_model(X, model, test_data, y, test_target): 

    # Evaluate training performance 
    y_train_pred = model.predict(X)
    train_mse = root_mean_squared_error(y, y_train_pred)

    print(f'Training MSE: {train_mse}')

    # Evaluate testing performance 
    y_test_pred = model.predict(test_data)
    test_mse = root_mean_squared_error(test_target, y_test_pred)

    print(f'Testing MSE: {test_mse}')

    # Log the MSE with mlflow
    mlflow.log_metric('train_mse', train_mse)
    mlflow.log_metric('test_mse', test_mse)

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

        
