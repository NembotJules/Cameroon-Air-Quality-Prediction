import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    page_title="Cameroon Air Quality",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .custom-metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #333;
    }
    .metric-subtitle {
        font-size: 0.7rem;
        color: #888;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data(date_str):
    """
    Load prediction data for a specific date from S3
    """
    # Replace this with your actual data loading logic
    # Example data generation for demonstration
    times = pd.date_range(start=f"{date_str} 00:00", end=f"{date_str} 23:59", freq='H')
    aqi_values = np.random.normal(15, 5, len(times))
    df = pd.DataFrame({
        'timestamp': times,
        'aqi': aqi_values
    })
    return df

def create_time_series(df):
    """Create a time series plot using Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['aqi'],
        mode='lines+markers',
        line=dict(color='#76B947', width=2),
        marker=dict(size=6, color='#76B947'),
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(230,230,230,0.8)',
            title=None,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(230,230,230,0.8)',
            title='AQI',
            range=[0, max(df['aqi']) * 1.2]
        ),
        height=300,
    )
    
    return fig

def create_map():
    """Create an interactive map using Plotly"""
    # Example coordinates for Cameroon cities
    cities = {
        'Yaoundé': {'lat': 3.8667, 'lon': 11.5167, 'aqi': 74},
        'Douala': {'lat': 4.0500, 'lon': 9.7000, 'aqi': 79},
        'Bamenda': {'lat': 5.9333, 'lon': 10.1667, 'aqi': 34},
    }
    
    fig = go.Figure()

    # Add scatter markers for cities
    fig.add_trace(go.Scattermapbox(
        lat=[cities[city]['lat'] for city in cities],
        lon=[cities[city]['lon'] for city in cities],
        mode='markers+text',
        marker=dict(
            size=15,
            color=[cities[city]['aqi'] for city in cities],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title='AQI'),
        ),
        text=[f"{city}: {cities[city]['aqi']}" for city in cities],
        textposition="top center",
        name='Cities'
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=7.3697, lon=12.3547),
            zoom=5
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
    )
    
    return fig

def main():
    # Header
    st.title("🌍 Cameroon Air Quality Dashboard")
    
    # City selector and date picker in the sidebar
    st.sidebar.title("Controls")
    selected_city = st.sidebar.selectbox(
        "Select City",
        ["Yaoundé", "Douala", "Bamenda"]
    )
    
    selected_date = st.sidebar.date_input(
        "Select Date",
        datetime.now().date()
    )
    
    # Load data
    df = load_data(selected_date)
    
    # Top metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="custom-metric-container">
                <div class="metric-title">Current AQI</div>
                <div class="metric-value">76</div>
                <div class="metric-subtitle">Updated 5 minutes ago</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="custom-metric-container">
                <div class="metric-title">24h Min</div>
                <div class="metric-value">11</div>
                <div class="metric-subtitle">at 10:09 AM</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="custom-metric-container">
                <div class="metric-title">24h Max</div>
                <div class="metric-value">24</div>
                <div class="metric-subtitle">at 2:08 PM</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Time series and map
    st.markdown("### 24-Hour AQI Trend")
    st.plotly_chart(create_time_series(df), use_container_width=True)
    
    st.markdown("### Regional Air Quality Map")
    st.plotly_chart(create_map(), use_container_width=True)
    
    # Additional information
    st.markdown("### Air Quality Information")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.info("""
        **PM2.5 Concentration (2023)**
        4.8 times the WHO annual air quality guideline value
        """)
    
    with info_col2:
        st.warning("""
        **Air Quality Status**
        Moderate - May cause breathing discomfort for sensitive groups
        """)

if __name__ == "__main__":
    main()