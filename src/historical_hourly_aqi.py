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
#cities
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
url = "https://air-quality-api.open-meteo.com/v1/air-quality"

for city in CITIES: 

    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "aerosol_optical_depth", "dust", "uv_index", "uv_index_clear_sky"],
        "timezone": "auto",
        "start_date": "2020-01-01",
        "end_date": "2024-01-01"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
    hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
    hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
    hourly_nitrogen_dioxide = hourly.Variables(3).ValuesAsNumpy()
    hourly_sulphur_dioxide = hourly.Variables(4).ValuesAsNumpy()
    hourly_ozone = hourly.Variables(5).ValuesAsNumpy()
    hourly_aerosol_optical_depth = hourly.Variables(6).ValuesAsNumpy()
    hourly_dust = hourly.Variables(7).ValuesAsNumpy()
    hourly_uv_index = hourly.Variables(8).ValuesAsNumpy()
    hourly_uv_index_clear_sky = hourly.Variables(9).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["pm10"] = hourly_pm10
    hourly_data["pm2_5"] = hourly_pm2_5
    hourly_data["carbon_monoxide"] = hourly_carbon_monoxide
    hourly_data["nitrogen_dioxide"] = hourly_nitrogen_dioxide
    hourly_data["sulphur_dioxide"] = hourly_sulphur_dioxide
    hourly_data["ozone"] = hourly_ozone
    hourly_data["aerosol_optical_depth"] = hourly_aerosol_optical_depth
    hourly_data["dust"] = hourly_dust
    hourly_data["uv_index"] = hourly_uv_index
    hourly_data["uv_index_clear_sky"] = hourly_uv_index_clear_sky

    hourly_aqi_dataframe = pd.DataFrame(data = hourly_data)
    hourly_aqi_dataframe.to_csv(f'hourly_aqi_{city["name"]}_dataframe')
