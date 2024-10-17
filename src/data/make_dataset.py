import pandas as pd
import numpy as np
import mlflow
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

default_config_name = "../../config/default.yaml"

with open(default_config_name, "r") as file: 
    default_config = yaml.safe_load(file)



def load_data(file_path):
    return pd.read_csv(file_path)


def clean_data(df):
    #drop unncessary columns(id, etc)

    if 'id' in df.columns: 
        df.drop('id', axis  = 1, inplace = True)

    #drop rows with missing values
    df.dropna(inplace = True)

    #drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df



def create_categorical_and_numeric_features(df):
    numeric_features = df.select_dtypes(include = ['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include = ['object']).columns
    return numeric_features, categorical_features



#Custom function for log transformation 
def log_transform(X, numeric_features):
    X[numeric_features] = np.log1p(X[numeric_features])
    return X

def label_encode(X, categorical_features):
    #Create a LabelEncoder for each categorical feature
    label_encoders = {col: LabelEncoder() for col in categorical_features}

    X = X.copy()
    for feature in categorical_features:
        X[feature] = label_encoders[feature].fit_transform(X[feature])
    return X

def to_dataframe(X, feature_names):
    return pd.DataFrame(X, columns = feature_names)


def preprocessor_transform(X,numeric_features, categorical_features):

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



def preprocess_data(df):

    if default_config["target_column"] in df.columns: 
        X = clean_data(df).drop(default_config["target_column"], axis = 1)
        y = clean_data(df)[default_config["target_column"]]

    else: 
        X = clean_data(df)
        y = None

    
    numeric_features, categorical_features = create_categorical_and_numeric_features(X)

    X = preprocessor_transform(X, numeric_features, categorical_features)

    return X, y


def save_data(df, file_path):

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




