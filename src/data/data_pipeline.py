from typing import List, Dict
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

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
weather_df_features = ["weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", 
               "apparent_temperature_min", "sunrise", "sunset", "daylight_duration", "sunshine_duration",
                 "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max",
                   "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"]


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
        combined_daily_df = pd.concat([combined_daily_df, daily_dataframe], axis = 0, ignore_index=True)
    
    # Save the combined DataFrame to CSV and return it
    combined_daily_df.to_csv('combined_daily_df.csv', index = False)
    return combined_daily_df


def create_aqi_df(aqi_url:str)-> pd.DataFrame: 
    pass



if __name__== "__main__": 
    create_weather_df(weather_url=weather_url, cities=CITIES, features=weather_df_features)