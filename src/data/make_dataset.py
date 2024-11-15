import pandas as pd
import numpy as np
import os
import yaml
from typing import Tuple, List, Optional, Dict
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



def identify_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify numeric and categorical features in the DataFrame with proper type checking.
    """
    # First, convert date column to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Initialize feature lists
    numeric_features = []
    categorical_features = []
    
    # Explicitly check each column
    for column in df.columns:
        # Skip date and city columns
        if column in ['date', 'city']:
            categorical_features.append(column)
            continue
            
        # Check if column contains numeric data
        try:
            pd.to_numeric(df[column])
            numeric_features.append(column)
        except (ValueError, TypeError):
            categorical_features.append(column)
    
    print(f"Identified numeric features: {numeric_features}")
    print(f"Identified categorical features: {categorical_features}")
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input DataFrame by handling data types and invalid data.
    """
    df_cleaned = df.copy()
    
    # Convert date column to datetime
    if 'date' in df_cleaned.columns:
        df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])

    if 'id' in df_cleaned.columns: 
        df_cleaned.drop('id', axis = 1, inplace=True)
        
    if 'sunset' in df_cleaned.columns: 
        df_cleaned.drop('sunset', axis = 1, inplace=True)

    if 'snowfall_sum' in df_cleaned.columns: 
        df_cleaned.drop('snowfall_sum', axis = 1, inplace=True)

    if 'sunrise' in df_cleaned.columns: 
        df_cleaned.drop('sunrise', axis = 1, inplace=True)

    
    
    # Ensure numeric columns are properly typed
    numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    # Remove missing values and duplicates
    df_cleaned = df_cleaned.dropna()
    df_cleaned = df_cleaned.drop_duplicates()
    
    print(f"Cleaned DataFrame shape: {df_cleaned.shape}")
    print(f"Column dtypes after cleaning:\n{df_cleaned.dtypes}")
    
    return df_cleaned


def transform_features(
    df: pd.DataFrame,
    feature_types: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Apply transformations to features based on their types with proper handling.
    """
    df_transformed = df.copy()
    
    # Handle numeric features
    numeric_features = feature_types['numeric']
    if numeric_features:
        # Apply StandardScaler to numeric features
        scaler = StandardScaler()
        df_transformed[numeric_features] = scaler.fit_transform(df_transformed[numeric_features])
        print(f"Transformed numeric features: {numeric_features}")
    
    # Handle categorical features (excluding 'date')
    categorical_features = [f for f in feature_types['categorical'] if f != 'date']
    if categorical_features:
        for feature in categorical_features:
            if feature == 'city':  # Special handling for city
                encoder = LabelEncoder()
                df_transformed[feature] = encoder.fit_transform(df_transformed[feature])
                print(f"Encoded categorical feature: {feature}")
    
    print(f"Transformed DataFrame shape: {df_transformed.shape}")
    print(f"Sample of transformed data:\n{df_transformed.head()}")
    
    return df_transformed


def preprocess_dataset_flow(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Main workflow for data preprocessing with improved flow control.
    """
    print("Starting preprocessing pipeline...")
    
    # Step 1: Clean the DataFrame
    print("\nStep 1: Cleaning DataFrame...")
    df_cleaned = clean_dataframe(df)
    
    # Step 2: Separate features and target if specified
    print("\nStep 2: Separating features and target...")
    X = df_cleaned.copy()
    y = None
    if target_column and target_column in df_cleaned.columns:
        y = df_cleaned[target_column]
        X = X.drop(columns=[target_column])
    
    # Step 3: Identify feature types
    print("\nStep 3: Identifying feature types...")
    feature_types = identify_feature_types(X)
    
    # Step 4: Transform features
    print("\nStep 4: Transforming features...")
    X_processed = transform_features(X, feature_types)
    
    # Save processed data
    print("\nSaving processed data...")
    X_processed.to_csv('processed_features.csv', index=False)
    if y is not None:
        y.to_csv('target.csv', index=False)
    
    print("\nPreprocessing pipeline completed successfully!")
    return X_processed, y



def save_data(df:pd.DataFrame, feature_save_path: str, target_save_path: str):

    X, y = preprocess_dataset_flow(df)

    if X.empty: 
        print(f"Warning: X is empty. No data will be saved.")
        return 

    try: 
    
        
        if y is not None: 
            if len(X) != len(y): 
                print(f"Warning: X and y have different lengths. No data will be saved.")
                return 
            #y.to_csv(default_config["data"]["preprocessed_train_target_path"], index = False)
            y.to_csv(target_save_path, index = False)
           
            print(" Train Target data saved successfully.")
            print(y.shape)

            #X.to_csv(default_config["data"]["preprocessed_train_data_path"], index=False)
            X.to_csv(feature_save_path, index = False)
            print("Training Feature data saved successfully.")
            print(X.shape)

        else: 
            X.to_csv(feature_save_path, index = False)
            print("Test Feature data saved successfully.")
            print(X.shape)



    except Exception as e: 
        print(f"Error saving data: {e}")


if __name__ == "__main__":


    df = load_data(default_config["data"]["raw_train_data_path"])
    test_data = load_data(default_config["data"]["raw_test_data_path"])
    save_data(df, default_config["data"]["preprocessed_train_data_path"], default_config["data"]["preprocessed_train_data_path"]) 
    save_data(test_data, default_config["data"]["preprocessed_test_data_path"], default_config["data"]["preprocessed_test_data_path"]) 




