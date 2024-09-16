import requests
import pandas as pd


# API endpoints and keys 


WEATHER_API_KEY = "42aff981a4c5edb672a02930373d5559"
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"
AIR_QUALITY_API_URL = "http://api.openweathermap.org/data/2.5/air_pollution"


# List of cities to analyze

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



@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def extract_weather_data(city): 

    """Extract weather data from OpenWeatherMap API"""

    params = {
        "lat": city["lat"], 
        "lon": city["lon"], 
        "appid": WEATHER_API_KEY, 
        "units": "metric"
    }

    response = requests.get(WEATHER_API_URL, params=params)

    if response.status_code == 200: 

        data = response.json()
    
        weather_data = {
            "city" : city["name"], 
            "temperature": data["main"]["temp"], 
            "feels_like": data["main"]["feels_like"], 
            "temp_min": data["main"]["temp_min"], 
            "temp_max": data["main"]["temp_max"], 
            "pressure": data["main"]["pressure"], 
            "humidity": data["main"]["humidity"], 
            "wind_speed": data["wind"]["speed"]
    }
        
        return weather_data
    else: 
        print(f"Failed to fetch data: {response.status_code}, {response.text}")
        return None
    

weather_data = extract_weather_data(CITIES[9])

if weather_data: 
    print(extract_weather_data(CITIES[9]))

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def extract_air_quality_data(city): 
    """ Extract air quality data from OpenWeatherMap API."""
    params = {
        "lat": city["lat"], 
        "lon": city["lon"], 
        "appid": WEATHER_API_KEY
    }

    response = requests.get(AIR_QUALITY_API_URL, params=params)

    if response.status_code == 200: 

        data = response.json()

        air_quality_data = {
            "city": city["name"], 
            "aqi": data["list"][0]["main"]["aqi"], 
            "co": data["list"][0]["components"]["co"], 
            "no2": data["list"][0]["components"]["no2"], 
            "o3": data["list"][0]["components"]["o3"], 
            "pm2_5": data["list"][0]["components"]["pm2_5"]
        }

        return air_quality_data
    
    else: 
        print(f"Failed to fetch data: {response.status_code}, {response.text}")


air_quality_data = extract_air_quality_data(CITIES[9])

if air_quality_data: 
    print(extract_air_quality_data(CITIES[9]))

@task   
def transform_data(weather_data, air_quality_data): 
    """ Transform and combine weather and air quality data."""

    weather_df = pd.DataFrame(weather_data)
    air_quality_df = pd.DataFrame(air_quality_data)

    combined_df = pd.merge(weather_df, air_quality_df, on = "city")

    #Add air quality category based on AQI

    combined_df["air_quality_category"] = pd.cut(
        combined_df["aqi"], 
        bins=[-1, 50, 100, 150, 200, 300, 500], 
        labels = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
    )

    return combined_df

@task
def load_data(df, db_path = "city_weather_air_quality.db"): 
    """ Load the data into SQLite database"""
    
    conn = sqlite3.connect(db_path)
    df.to_sql("city_data", conn, if_exists = "replace", index = False)
    conn.close()

@flow(name="City Weather and Air Quality ETL")
def city_weather_air_quality_etl(): 
    weather_data = []
    air_quality_data = []

    for city in CITIES: 
        weather_data.append(extract_weather_data(city))
        air_quality_data.append(extract_air_quality_data(city))

    combined_data = transform_data(weather_data, air_quality_data)
    load_data(combined_data)

    return combined_data

@task
def analyze_data(df): 
    """ Perform basic analysis on the collected data"""
    print("\nData Analysis: ")
    print(f"Total cities analyzed:  {len(df)}")
    print(f"\nAverage temperature: {df['temperature'].mean():.1f}°C")
    print(f"Average AQI: {df['aqi'].mean():.1f}")

    best_air_quality = df.loc[df['aqi'].idxmin()]
    worst_air_quality = df.loc[df['aqi'].idxmax()]

    print(f"\nCity with best air quality:  {best_air_quality['city']} (AQI: {best_air_quality['aqi']})")
    print(f"\nCity with worst air quality:  {worst_air_quality['city']} (AQI: {worst_air_quality['aqi']})")


@flow(name="City Data Analysis")
def city_data_analysis(): 
    data = city_weather_air_quality_etl()
    analyze_data(data)


if __name__ == "__main__": 
    city_data_analysis()