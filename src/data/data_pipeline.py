from typing import List, Dict, Optional
import openmeteo_requests
import requests_cache
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

import os
import pandas as pd
from retry_requests import retry
from prefect import flow, task

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

aqi_features = ["carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "aerosol_optical_depth", "dust", "uv_index", "uv_index_clear_sky"]

weather_df_features = ["weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", 
               "apparent_temperature_min", "sunrise", "sunset", "daylight_duration", "sunshine_duration",
                 "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max",
                   "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"]

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
            "apparent_temperature_max": daily.Variables(3).ValuesAsNumpy(),
            "apparent_temperature_min": daily.Variables(4).ValuesAsNumpy(),
            "sunrise": daily.Variables(5).ValuesAsNumpy(),
            "sunset": daily.Variables(6).ValuesAsNumpy(),
            "daylight_duration": daily.Variables(7).ValuesAsNumpy(),
            "sunshine_duration": daily.Variables(8).ValuesAsNumpy(),
            "precipitation_sum": daily.Variables(9).ValuesAsNumpy(),
            "rain_sum": daily.Variables(10).ValuesAsNumpy(),
            "snowfall_sum": daily.Variables(11).ValuesAsNumpy(),
            "precipitation_hours": daily.Variables(12).ValuesAsNumpy(),
            "wind_speed_10m_max": daily.Variables(13).ValuesAsNumpy(),
            "wind_gusts_10m_max": daily.Variables(14).ValuesAsNumpy(),
            "wind_direction_10m_dominant": daily.Variables(15).ValuesAsNumpy(),
            "shortwave_radiation_sum": daily.Variables(16).ValuesAsNumpy(),
            "et0_fao_evapotranspiration": daily.Variables(17).ValuesAsNumpy()
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
            "uv_index_clear_sky": hourly.Variables(7).ValuesAsNumpy()
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

    # Save the final DataFrame to CSV and return it
    combined_daily_aqi_df.to_csv('combined_daily_aqi_df.csv', index=False)
    return combined_daily_aqi_df



@task(log_prints=True)
def merge_aqi_weather_df(aqi_df_path: str, weather_df_path: str) -> Optional[pd.DataFrame]:
    """
    Combines weather_df and aqi_df on matching date and city rows.

    Args: 
        aqi_df_path (str): Path to the daily AQI DataFrame (CSV file with 'date' and 'city' columns).
        weather_df_path (str): Path to the daily weather DataFrame (CSV file with 'date' and 'city' columns).

    Returns: 
        pd.DataFrame: DataFrame containing daily weather and AQI data for all the cities,
        or None if files are not found or merging fails.
    """

    # Check if both files exist
    if not os.path.exists(aqi_df_path) or not os.path.exists(weather_df_path):
        print("Error: One or both of the input files were not found.")
        return None

    try:
        # Load data from the CSV files
        aqi_df = pd.read_csv(aqi_df_path)
        weather_df = pd.read_csv(weather_df_path)
        
        # Merge DataFrames on 'date' and 'city'
        combined_df = pd.merge(weather_df, aqi_df, on=['date', 'city', 'latitude', 'longitude'], how='inner')
        
        # Save combined DataFrame to CSV
        combined_df.to_csv('daily_weather_aqi_df.csv', index=False)
        
        return combined_df

    except Exception as e:
        print(f"Error occurred during merging: {e}")
        return None

@flow
def etl(): 
    create_weather_df(weather_url=weather_url, cities=CITIES, features=weather_df_features)
    create_aqi_df(aqi_url=aqi_url, cities=CITIES, features=aqi_features)
    merge_aqi_weather_df(aqi_df_path='combined_daily_aqi_df.csv', weather_df_path='combined_daily_weather_df.csv')

    



if __name__== "__main__": 
    etl()