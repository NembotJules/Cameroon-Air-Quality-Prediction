import pandas as pd
import numpy as np
import os
import yaml
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the config file
default_config_name = os.path.join(current_dir, '..', '..', 'config', 'default.yaml')

with open(default_config_name, "r") as file: 
    default_config = yaml.safe_load(file)



def load_data(file_path: str) -> str:
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    #drop unncessary columns(id, etc)

    if 'id' in df.columns: 
        df.drop('id', axis  = 1, inplace = True)

    #drop rows with missing values
    df.dropna(inplace = True)

    #drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df



def create_categorical_and_numeric_features(df: pd.DataFrame) -> Tuple[List[str], List[str]] :
    numeric_features = df.select_dtypes(include = ['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include = ['object']).columns
    return numeric_features.tolist(), categorical_features.tolist()



#Custom function for log transformation 
def log_transform(X:pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
    X[numeric_features] = np.log1p(X[numeric_features])
    return X

def label_encode(X:pd.DataFrame, categorical_features:List[str]) -> pd.DataFrame:
    #Create a LabelEncoder for each categorical feature
    label_encoders = {col: LabelEncoder() for col in categorical_features}

    X = X.copy()
    for feature in categorical_features:
        X[feature] = label_encoders[feature].fit_transform(X[feature])
    return X

def to_dataframe(X:np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame(X, columns = feature_names)


def preprocessor_transform(X: pd.DataFrame ,numeric_features:List[str], categorical_features: List[str]) -> pd.DataFrame:

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', 
         Pipeline(steps = [
             ('log', FunctionTransformer(lambda X: log_transform(X, numeric_features))), 
              ('scaler', StandardScaler()), 
         ]),
         numeric_features),
        ('cat', FunctionTransformer(lambda X: label_encode(X, categorical_features)), categorical_features)
    ], 
    remainder = 'passthrough'
    )

    #Fit and transform the training data
    X_transformed = preprocessor.fit_transform(X)
    print(len(X_transformed))
    X = to_dataframe(X_transformed, X.columns)
    

    return X



def preprocess_data(df:pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    print(f"The shape of the data at the beginning of preprocess_data {df.shape}")

    if default_config["target_column"] in df.columns: 
        X = clean_data(df).drop(default_config["target_column"], axis = 1)
        y = clean_data(df)[default_config["target_column"]]

    else: 
        X = clean_data(df)
        y = None

    print(f"The shape of the data after removing the id and the target column {X.shape}")

    
    numeric_features, categorical_features = create_categorical_and_numeric_features(X)

    X = preprocessor_transform(X, numeric_features, categorical_features)

    print(f"The shape of the data after the preprocessor_transform call {X.shape}")

    return X, y


def save_data(df:pd.DataFrame):

    X, y = preprocess_data(df)

    if X.empty: 
        print(f"Warning: X is empty. No data will be saved.")
        return 

    try: 
    
        
        if y is not None: 
            if len(X) != len(y): 
                print(f"Warning: X and y have different lengths. No data will be saved.")
                return 
            y.to_csv(default_config["data"]["preprocessed_train_target_path"], index = False)
           
            print(" Train Target data saved successfully.")
            print(y.shape)

            X.to_csv(default_config["data"]["preprocessed_train_data_path"], index=False)
            print("Training Feature data saved successfully.")
            print(X.shape)

        else: 
            X.to_csv(default_config["data"]["preprocessed_test_data_path"], index=False)
            print("Test Feature data saved successfully.")
            print(X.shape)



    except Exception as e: 
        print(f"Error saving data: {e}")


if __name__ == "__main__":


    df = load_data(default_config["data"]["raw_train_data_path"])
    test_data = load_data(default_config["data"]["raw_test_data_path"])
    save_data(df, default_config["data"]["preprocessed_train_data_path"]) 
    save_data(test_data, default_config["data"]["preprocessed_test_data_path"]) 




