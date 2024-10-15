import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer




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

    X = clean_data(df).drop('pm2_5', axis = 1)
    y = clean_data(df)['pm2_5']

    numeric_features, categorical_features = create_categorical_and_numeric_features(X)

    X = preprocessor_transform(X, numeric_features, categorical_features)

    return X, y


def split_data(df):

    #Load preprocessed data

    X, y = preprocess_data(df)
    # Split the data into training and testing sets
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, y_train, X_test, y_test


def save_data(df, file_path):

    X_train, y_train, X_test, y_test = split_data(df)
    print(X_train.head(5), X_test.head(5))


if __name__ == "__main__":
    df = load_data('../../data/train_test_data/train.csv')
    save_data(df, 'data/processed/train.csv') 



