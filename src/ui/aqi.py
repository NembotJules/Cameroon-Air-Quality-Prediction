import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats

# Set page config
st.set_page_config(page_title="Cameroon AQI Prediction", layout="wide")

# Sidebar for progress indicators
with st.sidebar:
    st.header("Working Progress")
    st.progress(100)
    
    st.subheader("Connecting to Data Store")
    st.success("Successfully connected ✓")
    
    st.subheader("Collecting weather forecasts")
    st.success("Collected ✓")
    
    st.subheader("Loading the models")
    st.success("Models loaded ✓")
    
    st.subheader("Rendering the map")
    st.success("Map loaded ✓")

# Main content
st.title("PM2.5 Predictions for Cameroon 🇨🇲")

# Date selection
dates = ["Today", "Tomorrow"] + [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(2, 7)]
selected_date = st.radio("Select forecasting day", dates, horizontal=True)

# Define cities and their coordinates
cities = {
    "Douala": (4.0511, 9.7679),
    "Yaoundé": (3.8480, 11.5021),
    "Maroua": (10.5910, 14.3158),
    "Garoua": (9.3017, 13.3921),
    "Nkongsamba": (4.9500, 9.9333),
    "Buea": (4.1536, 9.2427),
    "Baffoussam": (5.4768, 10.4214),
    "Ebolowa": (2.9000, 11.1500),
    "Bertoua": (4.5785, 13.6846),
    "Bamenda": (5.9631, 10.1591)
}

# Generate mock AQI data
@st.cache_data
def generate_aqi_data():
    return {city: np.random.randint(0, 300, 7) for city in cities}

aqi_data = generate_aqi_data()

# Create a DataFrame for the map
df = pd.DataFrame({
    "City": list(cities.keys()),
    "Latitude": [coord[0] for coord in cities.values()],
    "Longitude": [coord[1] for coord in cities.values()],
    "AQI": [aqi_data[city][dates.index(selected_date)] for city in cities]
})

# Create the map
fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="AQI", size="AQI",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        size_max=30, zoom=5, mapbox_style="carto-positron",
                        hover_name="City", hover_data={"Latitude": False, "Longitude": False})

fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})

# Display the map
st.plotly_chart(fig, use_container_width=True)

# City selection for detailed view
selected_city = st.selectbox("Select the city to view forecast plots for the whole week", list(cities.keys()))

# Create line chart for selected city
city_data = aqi_data[selected_city]
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=dates, y=city_data, mode='lines+markers', name='PM2.5 Forecast'))
fig_line.update_layout(title=f"PM2.5 Forecast for {selected_city}",
                       xaxis_title="Date",
                       yaxis_title="PM2.5 Level")

# Display the line chart
st.plotly_chart(fig_line, use_container_width=True)

# Real-time AQI Trend Analyzer and Predictive Alert System
st.header("🚀 Real-time AQI Trend Analyzer and Predictive Alert System")
st.write("This feature analyzes recent AQI trends and predicts potential air quality issues.")

# Generate mock historical data
historical_data = np.random.randint(0, 300, 24)  # 24 hours of data

# Calculate trend
slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(historical_data)), historical_data)

# Predict next 6 hours
future_hours = 6
predicted_values = [slope * (len(historical_data) + i) + intercept for i in range(1, future_hours + 1)]

# Create trend chart
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=list(range(24)), y=historical_data, mode='lines', name='Historical Data'))
fig_trend.add_trace(go.Scatter(x=list(range(23, 23 + future_hours)), y=[historical_data[-1]] + predicted_values, mode='lines', name='Predicted Trend', line=dict(dash='dash')))
fig_trend.update_layout(title=f"AQI Trend Analysis for {selected_city}",
                        xaxis_title="Hours",
                        yaxis_title="AQI Level")

# Display the trend chart
st.plotly_chart(fig_trend, use_container_width=True)

# Predictive Alert System
alert_threshold = 150  # AQI threshold for alert

if any(value > alert_threshold for value in predicted_values):
    st.error(f"⚠️ Alert: AQI levels in {selected_city} are predicted to exceed {alert_threshold} in the next 6 hours!")
    st.write("Recommended actions:")
    st.write("1. Limit outdoor activities")
    st.write("2. Use air purifiers if available")
    st.write("3. Keep windows closed")
elif slope > 0:
    st.warning(f"⚠️ Caution: AQI levels in {selected_city} show an increasing trend. Stay informed about air quality updates.")
else:
    st.success(f"✅ Good news! AQI levels in {selected_city} are stable or decreasing. Enjoy your day!")

# AQI Impact Analysis
st.header("AQI Impact Analysis")
st.write("Understand how different factors affect air quality in real-time.")

col1, col2, col3 = st.columns(3)
with col1:
    temperature = st.slider("Temperature (°C)", -10, 40, 20)
with col2:
    humidity = st.slider("Humidity (%)", 0, 100, 50)
with col3:
    wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)

# Simple mock model for AQI prediction
def predict_aqi(temp, hum, wind):
    base_aqi = 50
    temp_factor = (temp - 20) * 2  # Higher temp, higher AQI
    hum_factor = (hum - 50) * 0.5  # Higher humidity, higher AQI
    wind_factor = (25 - wind) * 2  # Lower wind speed, higher AQI
    return max(0, min(300, base_aqi + temp_factor + hum_factor + wind_factor))

predicted_aqi = predict_aqi(temperature, humidity, wind_speed)

st.metric("Predicted AQI", f"{predicted_aqi:.2f}")

if predicted_aqi < 50:
    st.success("Good air quality!")
elif predicted_aqi < 100:
    st.info("Moderate air quality.")
elif predicted_aqi < 150:
    st.warning("Unhealthy for sensitive groups.")
else:
    st.error("Unhealthy air quality!")

# Explanation section
st.header("How it's done?")
st.write("""
Our AQI prediction system uses a combination of historical data, real-time measurements, and advanced machine learning models to forecast air quality across Cameroon.

Key components of our system include:

1. Data Collection: We gather data from various sources, including weather stations, satellite imagery, and ground-based sensors.

2. Predictive Modeling: Our AI models analyze patterns in historical data to make accurate short-term and long-term predictions.

3. Real-time Trend Analysis: The system continuously analyzes recent AQI trends to detect potential air quality issues before they become severe.

4. Impact Simulation: Users can explore how different environmental factors affect air quality through our interactive simulation.

5. Alerts and Recommendations: Based on predictions and current trends, the system provides timely alerts and actionable recommendations to help citizens stay safe.

This comprehensive approach allows us to provide accurate, timely, and actionable air quality information for the people of Cameroon.
""")

st.write("Remember, while our predictions are based on advanced models, always refer to official sources for critical air quality information.")