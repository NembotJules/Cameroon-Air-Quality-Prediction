from typing import List, Dict, Optional, Tuple
import json
import openmeteo_requests
import requests
import requests_cache
import pandas as pd
import os
import io
from retry_requests import retry
from prefect_aws.s3 import S3Bucket
from prefect import flow, task
from prefect_github import GitHubCredentials
from prefect.runner.storage import GitRepository
from prefect.blocks.system import Secret
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the config file
default_config_name = os.path.join(current_dir, '..', '..', 'config', 'default.yaml')

with open(default_config_name, "r") as file: 
    default_config = yaml.safe_load(file)



# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
CITIES = [

   {"name": "Douala", "lat": 4.0483, "lon": 9.7043}, 
   {"name": "Yaoundé", "lat": 3.8667, "lon": 11.5167}, 
   {"name": "Bafoussam", "lat": 5.4737, "lon": 10.4179}, 
   {"name": "Bamenda", "lat": 5.9527, "lon": 10.1582}, 
   {"name": "Maroua", "lat": 10.591, "lon": 14.3159}, 
   {"name": "Ngaoundéré", "lat": 7.3167, "lon": 13.5833}, 
   {"name": "Buea", "lat": 4.1527, "lon": 9.241}, 
   {"name": "Ebolowa", "lat": 2.9, "lon": 11.15}, 
   {"name": "Garoua", "lat": 9.3, "lon": 13.4}, 
   {"name": "Bertoua", "lat": 4.5833, "lon": 13.6833}, 

]

weather_url = "https://api.open-meteo.com/v1/forecast"
aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

aqi_features = ["carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "aerosol_optical_depth", "dust", "uv_index", "uv_index_clear_sky", "pm2_5"]

weather_df_features = ["weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "apparent_temperature_max", 
               "apparent_temperature_min", "apparent_temperature_mean", "daylight_duration", "sunshine_duration",
                 "precipitation_sum", "rain_sum",  "precipitation_hours", "wind_speed_10m_max",
                   "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"]

s3_bucket_block = S3Bucket.load("cameroon-air-quality-bucket")

def read_csv_from_s3(s3_path):

    """
    Read csv file path from AWS S3 and return the corresponding Dataframe

    Args: 
        s3_path: the path to the csv file


    Returns:
        type_: pd.DataFrame
    """
    # Extract bucket and key from the full S3 path
    bucket, key = s3_path.replace('s3://', '').split('/', 1)
    
    # Create a BytesIO object to store the downloaded file
    file_object = io.BytesIO()
    
    # Download the file to the BytesIO object
    s3_bucket_block.download_object_to_file_object(
        from_path=key,  # The key/path of the file in the S3 bucket
        to_file_object=file_object  # The file-like object to write to
    )
    
    # Reset the file pointer to the beginning
    file_object.seek(0)
    
    # Read the CSV
    return pd.read_csv(file_object)


def upload_df_to_s3(df, s3_path):
    """
    Upload a pandas DataFrame to S3 as a CSV.
    
    Args:
    - df (pandas.DataFrame): DataFrame to upload
    - s3_path (str): Full S3 path where the file will be uploaded 
                     (e.g., 's3://cameroon-air-quality-bucket/data/output/my_file.csv')
    """
    # Create a BytesIO buffer
    csv_buffer = io.BytesIO()
    
    # Write the DataFrame to the buffer as CSV
    df.to_csv(csv_buffer, index=False)
    
    # Reset the buffer position to the beginning
    csv_buffer.seek(0)
    
    # Extract the key (path within the bucket)
    key = s3_path.replace('s3://', '').split('/', 1)[1]
    
    # Upload the file object to S3
    s3_bucket_block.upload_from_file_object(
        csv_buffer, 
        to_path=key
    )
    
    print(f"Successfully uploaded to {s3_path}")


@task(log_prints=True)
def create_weather_df(weather_url:str, cities: List[Dict[str, float]], features: List[str]) -> pd.DataFrame: 
    """
    Fetches weather data for each city, processes it, and combines it into a DataFrame.

    
    Args: 
        weather_url (str): The API URL for fetching weather data.
        cities(List[Dict[str, float]]): A list of dictionaries with each city's name, latitude, longitude.
        features(List[str]): A list of the features  we want to extract.

    Returns: 
        pd.DataFrame: Combined DataFrame containing daily weather data for all cities.
    
    """

    combined_daily_df = pd.DataFrame(columns = weather_df_features)
    for city in cities: 

        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "daily": features,
            "timezone": "auto"
        }

        try: 
            responses = openmeteo.weather_api(weather_url, params=params)
            response = responses[0]

        except Exception as e: 
            print(f"Failed to fetch data for {city['name']}: {e}")
            continue

        print(f"Fetching data for {city['name']} at {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        # Process daily data. The order of variables needs to be the same as requested.
        daily = response.Daily()
        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "city": city["name"],
            "latitude": city["lat"], 
            "longitude": city["lon"],
            "weather_code": daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_max": daily.Variables(1).ValuesAsNumpy(),
            "temperature_2m_min": daily.Variables(2).ValuesAsNumpy(),
            "temperature_2m_mean": daily.Variables(3).ValuesAsNumpy(),
            "apparent_temperature_max": daily.Variables(4).ValuesAsNumpy(),
            "apparent_temperature_min": daily.Variables(5).ValuesAsNumpy(),
            "apparent_temperature_mean": daily.Variables(6).ValuesAsNumpy(),
            "daylight_duration": daily.Variables(7).ValuesAsNumpy(),
            "sunshine_duration": daily.Variables(8).ValuesAsNumpy(),
            "precipitation_sum": daily.Variables(9).ValuesAsNumpy(),
            "rain_sum": daily.Variables(10).ValuesAsNumpy(),
            "precipitation_hours": daily.Variables(11).ValuesAsNumpy(),
            "wind_speed_10m_max": daily.Variables(12).ValuesAsNumpy(),
            "wind_gusts_10m_max": daily.Variables(13).ValuesAsNumpy(),
            "wind_direction_10m_dominant": daily.Variables(14).ValuesAsNumpy(),
            "shortwave_radiation_sum": daily.Variables(15).ValuesAsNumpy(),
            "et0_fao_evapotranspiration": daily.Variables(16).ValuesAsNumpy()
        }

        #Create a DataFrame for the current city's data and concatenate

        daily_dataframe = pd.DataFrame(data = daily_data)
        daily_dataframe['date'] = daily_dataframe['date'].dt.date
        combined_daily_df = pd.concat([combined_daily_df, daily_dataframe], axis = 0, ignore_index=True)
    
    # Save the combined DataFrame to CSV and return it
    print(f"The shape of the daily weather dataframe is {combined_daily_df.shape}")
    combined_daily_df.to_csv('combined_daily_weather_df.csv', index = False)
    return combined_daily_df

@task(log_prints=True)
def create_aqi_df(aqi_url:str, cities: List[Dict[str, float]], features: List[str])-> pd.DataFrame: 
    """
    Fetches AQI data for each city, processes hourly data to daily averages, and combines it into a final DataFrame.

    Args: 
        aqi_url(str): The API URL for fetching AQI data.
        cities(List[Dict[str, float]]): A list of dicitionaries with each city's name, latitude, and longitude.
        features (List[str]): List of AQI features to retrieve hourly data for.

    Retuns: 
        pd.DataFrame: Combined DataFrame containing daily AQI data (07 days) for all cities.

    """

    # Initialize an empty DataFrame for the final daily data...
    combined_daily_aqi_df = pd.DataFrame()

    for city in cities:
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "hourly": features,
            "timezone": "auto",
            "forecast_days": 6
        }

        try:
            responses = openmeteo.weather_api(aqi_url, params=params)
            response = responses[0]  # Assuming a single response is expected per city
        except Exception as e:
            print(f"Failed to fetch data for {city['name']}: {e}")
            continue

        print(f"Fetching AQI data for {city['name']} at {response.Latitude()}°N {response.Longitude()}°E")

        # Process hourly data
        hourly = response.Hourly()
        hourly_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "city": city["name"],
            "latitude": city["lat"], 
            "longitude": city["lon"],
            "carbon_monoxide": hourly.Variables(0).ValuesAsNumpy(),
            "nitrogen_dioxide": hourly.Variables(1).ValuesAsNumpy(),
            "sulphur_dioxide": hourly.Variables(2).ValuesAsNumpy(),
            "ozone": hourly.Variables(3).ValuesAsNumpy(),
            "aerosol_optical_depth": hourly.Variables(4).ValuesAsNumpy(),
            "dust": hourly.Variables(5).ValuesAsNumpy(),
            "uv_index": hourly.Variables(6).ValuesAsNumpy(),
            "uv_index_clear_sky": hourly.Variables(7).ValuesAsNumpy(), 
            "pm2_5": hourly.Variables(8).ValuesAsNumpy()
        }

        hourly_df = pd.DataFrame(data=hourly_data)

        # Convert hourly data to daily averages...
        hourly_df['date'] = hourly_df['datetime'].dt.date
        daily_df = hourly_df.groupby('date').mean(numeric_only=True).reset_index()
        

        # Impute missing values with forward and back fill...
        daily_df.ffill(inplace=True)
        daily_df.bfill(inplace=True)
        
        # Assign the city name to each row in daily data...
        daily_df['city'] = city["name"]

        # Combine each city's daily data
        combined_daily_aqi_df = pd.concat([combined_daily_aqi_df, daily_df], ignore_index=True)

    # Ensure the final dataset has 70 rows (7 days x 10 cities)
    print(f"The shape of the combined daily AQI Dataframe is {combined_daily_aqi_df.shape}")
    assert combined_daily_aqi_df.shape[0] == 70, "Final dataset does not have 70 rows as expected."

    #Saving the target column...
    y_pipeline = combined_daily_aqi_df[default_config["target_column"]]
    y_pipeline.to_csv('y_pipeline.csv', index = False)

    upload_df_to_s3(y_pipeline, default_config["data"]["preprocessed_pipeline_target_path"])

    
    
    #y_pipeline.to_csv(default_config["data"]["preprocessed_pipeline_target_path"])

    assert y_pipeline.shape[0] == 70, "Target dataset fetched from the API does not have 70 rows as expected"

    print("Successfully fetched and saved current PM 2_5 values (target) from the API")

    combined_daily_aqi_df.drop(default_config["target_column"], axis = 1, inplace = True)

    # Save the final DataFrame to CSV and return it
    combined_daily_aqi_df.to_csv('combined_daily_aqi_df.csv', index=False)
    return combined_daily_aqi_df

@task(log_prints=True)
def merge_aqi_weather_df(aqi_df: pd.DataFrame, weather_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Combines weather_df and aqi_df on matching date and city rows with comprehensive validation.
    
    Args:
        aqi_df (pd.DataFrame): DataFrame containing AQI data
        weather_df (pd.DataFrame): DataFrame containing weather data
        
    Returns:
        Optional[pd.DataFrame]: Merged DataFrame if successful, None if critical errors occur
    """
    try:
        # Store original row counts for validation
        original_counts = {
            'weather': len(weather_df),
            'aqi': len(aqi_df)
        }
        
        print("\n=== Pre-Processing Validation ===")
        
        # Check for missing values in key columns
        key_columns = ['date', 'city', 'latitude', 'longitude']
        for df_name, df in [('Weather', weather_df), ('AQI', aqi_df)]:
            missing = df[key_columns].isnull().sum()
            if missing.any():
                print(f"\nWarning: Missing values found in {df_name} DataFrame:")
                print(missing[missing > 0])
        
        # Clean and standardize the data
        for df in [weather_df, aqi_df]:
            # Convert dates to consistent format
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Clean string columns
            df['city'] = df['city'].str.strip()
            
            # Round coordinates for consistency
            df['latitude'] = df['longitude'].round(4)
            df['longitude'] = df['longitude'].round(4)
        
        print("\n=== Data Coverage Analysis ===")
        
        # Get unique cities
        all_cities = sorted(set(weather_df['city'].unique()) | set(aqi_df['city'].unique()))
        
        # Compare data coverage for each city
        coverage_issues = False
        for city in all_cities:
            weather_dates = set(weather_df[weather_df['city'] == city]['date'])
            aqi_dates = set(aqi_df[aqi_df['city'] == city]['date'])
            
            weather_count = len(weather_dates)
            aqi_count = len(aqi_dates)
            
            print(f"\nCity: {city}")
            print(f"Weather data dates: {weather_count}")
            print(f"AQI data dates: {aqi_count}")
            
            # Check for mismatches
            missing_in_weather = aqi_dates - weather_dates
            missing_in_aqi = weather_dates - aqi_dates
            
            if missing_in_weather:
                coverage_issues = True
                print(f"Dates missing in weather data: {sorted(missing_in_weather)}")
            if missing_in_aqi:
                coverage_issues = True
                print(f"Dates missing in AQI data: {sorted(missing_in_aqi)}")
            
            # Check coordinate consistency
            weather_coords = weather_df[weather_df['city'] == city][['latitude', 'longitude']].drop_duplicates()
            aqi_coords = aqi_df[aqi_df['city'] == city][['latitude', 'longitude']].drop_duplicates()
            
            if len(weather_coords) > 1 or len(aqi_coords) > 1:
                coverage_issues = True
                print(f"Warning: Multiple coordinate pairs found for {city}")
                print("Weather coordinates:")
                print(weather_coords)
                print("AQI coordinates:")
                print(aqi_coords)
        
        # Perform the merge
        print("\n=== Performing Merge ===")
        combined_df = pd.merge(
            weather_df,
            aqi_df,
            on=['date', 'city', 'latitude', 'longitude'],
            how='inner',
            validate='1:1'  # Ensure we don't get duplicate matches
        )
        
        # Post-merge validation
        print("\n=== Post-Merge Validation ===")
        
        # Check for expected number of rows per city
        city_stats = []
        for city in all_cities:
            weather_city_count = len(weather_df[weather_df['city'] == city])
            aqi_city_count = len(aqi_df[aqi_df['city'] == city])
            merged_city_count = len(combined_df[combined_df['city'] == city])
            
            expected_count = min(weather_city_count, aqi_city_count)
            if merged_city_count != expected_count:
                print(f"\nWarning: Unexpected row count for {city}")
                print(f"Expected: {expected_count}, Got: {merged_city_count}")
            
            city_stats.append({
                'city': city,
                'weather_rows': weather_city_count,
                'aqi_rows': aqi_city_count,
                'merged_rows': merged_city_count
            })
        
        # Create a summary DataFrame
        summary_df = pd.DataFrame(city_stats)
        print("\n=== Merge Summary ===")
        print(summary_df.to_string())
        
        # Final validation
        total_merged = len(combined_df)
        print(f"\nTotal rows in merged dataset: {total_merged}")
        if coverage_issues:
            print("\nWarning: Some data coverage issues were detected. Please review the logs above.")
        
        # Save combined DataFrame to CSV
        output_path = 'daily_weather_aqi_df.csv'
        combined_df.to_csv(output_path, index=False)
        print(f"\nMerged data saved to: {output_path}")
        
        return combined_df

    except Exception as e:
        print(f"Error occurred during merging: {e}")
        print("Stack trace:")
        import traceback
        print(traceback.format_exc())
        return None



@task
def create_date_city_df(df: pd.DataFrame) -> pd.DataFrame: 
    date_city_df = df[['date', 'city']]
    df.drop('date', axis = 1, inplace=True)
    return date_city_df

@task(log_prints=True)
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

@task(log_prints=True)
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input DataFrame by handling data types and invalid data.
    """
    df_cleaned = df.copy()
    
    # Convert date column to datetime
    if 'date' in df_cleaned.columns:
        df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
    
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

@task(log_prints=True)
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

@flow(name="Data Preprocessing Pipeline")
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


@task
def save_processed_data(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    save_path: Optional[str]
) -> None:
    """
    Save processed features, add them to the historical processed features to enable retraining later, and
    save the final result
    """
    if not save_path:
        return
    
    try:
        # Save current processed features
        X.to_csv(save_path, index=False)
        upload_df_to_s3(X, save_path)

        #Historical data paths
        historical_features_path = default_config['data']['preprocessed_train_data_path']
        historical_target_path = default_config['data']['preprocessed_train_target_path']

        # Historical data features (X, y)
        historical_features_df = read_csv_from_s3(historical_features_path)
        historical_target_df = read_csv_from_s3(historical_target_path)

        # Safely drop 'Unnamed: 0' column if it exists
        for df in [historical_features_df, historical_target_df]:
            if 'Unnamed: 0' in df.columns:
                df.drop('Unnamed: 0', axis=1, inplace=True)

        # Trying to concatenate historical features df with the current fetch features to enable retraining later...
        try:
            historical_features_df = pd.concat(
                [historical_features_df, X],
                axis=0,
                ignore_index=True
            )

            upload_df_to_s3(historical_features_df, historical_features_path)
        except Exception as e:
            raise Exception(
                "Failed to concatenate current and historical features: "
                f"{str(e)}"
            )
        
        # Trying to load pm2_5 values (target) from the actual pipeline run
        try: 
            y_pipeline = read_csv_from_s3(default_config["data"]["preprocessed_pipeline_target_path"])
        except Exception as e: 
            raise Exception(f"Failed to read pipeline target values: {str(e)}")

        try:
            historical_target_df = pd.concat(
                [historical_target_df, y_pipeline],
                axis=0,
                ignore_index=True
            )

            upload_df_to_s3(historical_target_df, historical_target_path)
            
        except Exception as e:
            raise Exception(
                "Failed to concatenate current and historical target values: "
                f"{str(e)}"
            )
        

        # Save target variable if provided
        if y is not None:
            target_path = f"{save_path.rsplit('.', 1)[0]}_target.csv"
            upload_df_to_s3(y, target_path)
            
    except Exception as e:
        raise Exception(f"Failed to save processed data: {str(e)}")

@task(log_prints=True)
def send_to_model_api(features_df: pd.DataFrame, api_url: str) -> List[float]:
    """
    Sends preprocessed features to the model API and gets predictions.
    
    Args:
        features_df: Preprocessed features DataFrame
        api_url: URL of the model API endpoint
    
    Returns:
        List of predictions
    """
    try:
        # Convert DataFrame to dictionary where each column becomes a list
        # This maintains the exact structure expected by the API
        features_dict = {
            column: features_df[column].tolist()
            for column in features_df.columns
        }
        
        # Create the payload structure exactly as expected by the API
        payload = {
            'features': features_dict
        }

        # Print the payload for debugging
        print("Sending payload:", json.dumps(payload, indent=2))
        
        # Send POST request to model API
        response = requests.post(
            api_url,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        
        
        response.raise_for_status()
        
        # Extract predictions from response
        predictions = response.json().get('predictions', [])
        print(f"Successfully received {len(predictions)} predictions from model API")
        return predictions
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling model API: {str(e)}")
        # Print response content for debugging if available
        if hasattr(e.response, 'content'):
            print(f"Response content: {e.response.content}")
        raise


@task(log_prints=True)
def create_predictions_df(predictions: List[float], date_city_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame combining predictions with their corresponding dates and cities.
    
    Args:
        predictions: List of model predictions
        date_city_df: DataFrame containing dates and cities
    
    Returns:
        DataFrame with predictions, dates, and cities
    """
    if len(predictions) != len(date_city_df):
        raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match number of rows in date_city_df ({len(date_city_df)})")
    
    predictions_df = date_city_df.copy()
    predictions_df['prediction'] = predictions
    
    return predictions_df

@task(log_prints=True)
def save_predictions(predictions_df: pd.DataFrame, base_output_path: str) -> None:
    """
    Saves the predictions DataFrame to CSV files organized by date and city.
    
    Args:
        predictions_df: DataFrame containing predictions with columns: date, city, prediction
        base_output_path: Base S3 path where to save the CSV files
                         (e.g., 's3://cameroon-air-quality-bucket/data/pipeline_output/predictions')
    
    The function will create the following structure:
    base_output_path/
        ├── YYYY-MM-DD/
        │   ├── city1.csv
        │   ├── city2.csv
        │   └── ...
        └── ...
    """
    try: 

        # s3_bucket_block.upload_from_dataframe(
        #         dataframe=predictions_df, 
        #         to_path= default_config['data']['prediction_dataframe_short_path'], 
        #         index = False
        #     )

        # I also want to saved the complete prediction dataframe
       # predictions_df.to_csv(default_config["data"]["prediction_dataframe_path"], index= False)
        upload_df_to_s3(predictions_df, default_config["data"]["prediction_dataframe_path"])
        # Ensure date column is datetime type
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        
        # Get unique dates
        unique_dates = predictions_df['date'].dt.strftime('%Y-%m-%d').unique()
        
        # Create directories and save files for each date
        for date in unique_dates:
            # Create the date directory path
            date_dir = f"{base_output_path}/{date}"
            
            # Get predictions for this date
            date_predictions = predictions_df[
                predictions_df['date'].dt.strftime('%Y-%m-%d') == date
            ]
            
            # Get unique cities for this date
            cities = date_predictions['city'].unique()
            
            # Save predictions for each city
            for city in cities:
                # Get city-specific predictions
                city_predictions = date_predictions[
                    date_predictions['city'] == city
                ].copy()
                
                # Create the full output path for this city
                city_output_path = f"{date_dir}/{city.lower()}.csv"
                
                # Save the predictions

            #     s3_bucket_block.upload_from_dataframe(
            #     dataframe=city_predictions, 
            #     to_path= city_output_path, 
            #     index = False
            # )

               # city_predictions.to_csv(city_output_path, index=False)
                upload_df_to_s3(city_predictions, city_output_path)
                print(f"Successfully saved predictions for {city} on {date} to {city_output_path}")
        
        print(f"Successfully saved all predictions to {base_output_path}")
        
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        raise



@flow(name= "Air Quality Pipeline",log_prints=True)
def main_flow(): 
    AQI_API_URL = "http://18.209.19.207:8000/predict"
    PREDICTIONS_OUTPUT_PATH = "predictions.csv"

    # date_city_df, predictions_df = predict_and_save(AQI_API_URL, PREDICTIONS_OUTPUT_PATH)

    weather_df = create_weather_df(weather_url=weather_url, cities=CITIES, features=weather_df_features)
    aqi_df = create_aqi_df(aqi_url=aqi_url, cities=CITIES, features=aqi_features)
    merged_df = merge_aqi_weather_df(weather_df=weather_df, aqi_df=aqi_df)
    date_city_df = create_date_city_df(merged_df)
     # Run the workflow
    X_processed, y = preprocess_dataset_flow(
        merged_df,
       # target_column='target',
    )

    save_processed_data(X_processed, y, save_path= default_config['data']['preprocessed_pipeline_features_data_path'])
    predictions = send_to_model_api(X_processed, AQI_API_URL)
    predictions_df = create_predictions_df(predictions, date_city_df=date_city_df)
    save_predictions(predictions_df, base_output_path=default_config["data"]["predictions_base_output_path"])





if __name__ == "__main__":
    

    main_flow()
 
    # main_flow.from_source(
        
    #      source=GitRepository(
    #         url="https://github.com/NembotJules/Cameroon-Air-Quality-Prediction.git",
    #         branch="dev",
    #         credentials=GitHubCredentials.load("git-credentials")
    #         ),
    #     entrypoint = "src/data/data_pipeline.py:main_flow"
    # ).deploy(
    #     name="air-quality-pipeline-managed-2", 
    #      work_pool_name="Managed-Pool", 
    #  )
    # main_flow.from_source(
    #     source="https://github.com/NembotJules/Cameroon-Air-Quality-Prediction.git",
    #     entrypoint="src/data/data_pipeline.py:main_flow")



    # ).deploy(
    #     name="Air-Quality-Pipeline-Deployment",
    #     work_pool_name="docker-test",
    #     image=DockerImage(
    #         name="985539786581.dkr.ecr.us-east-1.amazonaws.com/prefect-flows",
    #         platform="linux/amd64"
    #     ),

    #     schedules = [
    #         CronSchedule(
    #             cron = "58 05 * * *",
    #             timezone="Africa/Douala"
    #         )
    #     ], 
        
    # )
    

