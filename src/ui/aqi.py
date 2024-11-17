import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor
import glob
import yaml

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the config file
default_config_name = os.path.join(current_dir, '..', '..', 'config', 'default.yaml')

with open(default_config_name, "r") as file: 
    default_config = yaml.safe_load(file)


# Cache the data loading functions to improve performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_prediction_for_city_date(base_path: str, city: str, date: str) -> pd.DataFrame:
    """
    Load prediction data for a specific city and date
    """
    try:
        file_path = f"{base_path}/{date}/{city.lower()}.csv"
        if file_path:
            df = pd.read_csv(file_path)
           # df.to_csv('df.csv', index=False)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading prediction: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_all_cities_latest_predictions(base_path: str, date: str) -> pd.DataFrame:
    """
    Load predictions for all cities for a specific date
    """
    cities = ['Douala', 'Yaoundé', 'Bafoussam', 'Bamenda', 'Buea', 'Ngaoundéré', 'Ebolowa', 'Maroua', 'Bertoua', 'Garoua']
    data = []
    
    def load_city_data(city):
        city_data = load_prediction_for_city_date(base_path, city, date)
        if city_data is not None:
            return {
                'city': city,
                'PM2.5': city_data['prediction'].iloc[0]
            }
        return None
    
    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(load_city_data, cities))
    
    data = [result for result in results if result is not None]
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def load_forecast_data(base_path: str, city: str, start_date: datetime) -> pd.DataFrame:
    """
    Efficiently load forecast data for a city across 7 days
    """
    forecast_data = []
    dates = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(7)]
    
    def load_date_data(date):
        df = load_prediction_for_city_date(base_path, city, date)
        if df is not None:
            return {
                'date': date,
                'PM2.5': df['prediction'].iloc[0]
            }
        return None
    
    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor(max_workers=7) as executor:
        results = list(executor.map(load_date_data, dates))
    
    forecast_data = [result for result in results if result is not None]
    return pd.DataFrame(forecast_data)

def get_aqi_recommendation(aqi: float) -> str:
    """
    Get recommendation based on AQI value
    """
    if aqi <= 50:
        return "Good air quality. Ideal for outdoor activities."
    elif aqi <= 100:
        return "Moderate air quality. Sensitive groups should reduce prolonged outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy for sensitive groups. Consider reducing outdoor activities."
    else:
        return "Unhealthy air quality. Avoid prolonged outdoor activities."

def calculate_aqi_from_pm25(pm25: float) -> float:
    """
    Approximate AQI from pm25(simplified conversion)
    """
    return pm25 * 4

def initialize_session_state():
    """
    Initialize session state variables
    """
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

def load_initial_data(base_path: str):
    """
    Load all initial data needed for the application
    """
    with st.spinner('Initializing Air Quality Prediction System...'):
        # Get today's date and calculate date range
        today = datetime.now().date()
        
        # Load initial data for all cities
        initial_data = load_all_cities_latest_predictions(base_path, today.strftime('%Y-%m-%d'))
        
        if initial_data is not None:
            st.session_state.data_loaded = True
            return True
    return False

def main():
    st.set_page_config(page_title="Air Quality Prediction System", layout="wide")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for loading status and information
    with st.sidebar:
        st.title("System Status")
        base_path = default_config["data"]["predictions_base_output_path"] # Replace with your actual base path
        
        if not st.session_state.data_loaded:
            if load_initial_data(base_path):
                st.success("✅ System initialized successfully")
            else:
                st.error("❌ Error initializing system")
                return
        
        st.info("ℹ️ System Information")
        #st.write("• Data updates every hour")
        st.write("• Predictions available for 5 days")
        st.write("• Currently monitoring 10 cities")
    
    # Only show main UI if data is loaded
    if st.session_state.data_loaded:
        # Title with icon
        st.title("🌬️ Air Quality Prediction System")
        
        # Create two columns for the main layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Air Quality Prediction for Cameroon")
            st.text("Select a city and date to view predictions")
            
            # City selection
            cities = ['Douala', 'Yaoundé', 'Bafoussam', 'Bamenda', 'Buea', 'Ngaoundéré', 'Ebolowa', 'Maroua', 'Bertoua', 'Garoua']
            selected_city = st.selectbox("", cities)
            
            # Date selection with 7-day limit
            today = datetime.now().date()
            max_date = today + timedelta(days=5)
            selected_date = st.date_input(
                "Select a date",
                today,
                min_value=today,
                max_value=max_date
            )
            date_str = selected_date.strftime('%Y-%m-%d')
            
            # Load prediction data
            prediction_data = load_prediction_for_city_date(base_path, selected_city, date_str)
            
            if prediction_data is not None:
                pm25 = prediction_data['prediction'].iloc[0]
                aqi = calculate_aqi_from_pm25(pm25)
                
                # Display metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Air Quality Index (AQI)", f"{aqi:.0f}")
                with metrics_col2:
                    st.metric("PM2.5 Level", f"{pm25:.1f} µg/m³")
                
                # Display recommendation
                st.subheader("Recommendation")
                st.info(get_aqi_recommendation(aqi))
                
                # 7-Day forecast
                st.subheader(f"5-Day AQI Forecast for {selected_city}")
                
                # Load actual forecast data
                forecast_data = load_forecast_data(base_path, selected_city, today)
                
                if not forecast_data.empty:
                    fig = px.line(forecast_data, x='date', y='PM2.5',
                                title=f"AQI Forecast - {selected_city}")
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="AQI",
                        hovermode='x'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("City Comparison")
            st.text("AQI across cities")
            
            # Load comparison data
            comparison_data = load_all_cities_latest_predictions(base_path, date_str)
            
            if comparison_data is not None:
                
                fig = px.bar(comparison_data, x='city', y='PM2.5',
                            title="AQI Comparison Across Cities")
                fig.update_layout(
                    xaxis_title="City",
                    yaxis_title="AQI",
                    hovermode='x'
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

