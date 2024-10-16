import mlflow
import pandas as pd


best_run_id = '06c745047aa74cc3abdf15de9d58704d'
best_model_name = 'XGBR2'

mlflow.set_tracking_uri('http://35.153.184.244:5000/')
mlflow.set_experiment('air-quality-prediction')

def train_model(X, y): 

    logged_model = f'runs:/{best_run_id}/{best_model_name}'

    # Load model as a PyFuncModel.
    XGBR2 = mlflow.xgboost.load_model(logged_model)

    # Train the model on the full dataset
    XGBR2.fit(X, y) 

    # Log the model with mlflow
    mlflow.xgboost.log_model(XGBR2, artifact_path = best_model_name)

    return XGBR2


if __name__ == "__main__": 

    X = pd.read_csv('../../data/train_test_data/X.csv')
    y = pd.read_csv('../../data/train_test_data/y.csv')

    train_model(X, y)

