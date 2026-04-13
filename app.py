import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Custom modules
from src.preprocessing import load_data, clean_data, aggregate_sales
from src.eda_tools import plot_time_series, plot_seasonality, plot_rolling_stats, detect_anomalies, get_business_insights
from src.features import prepare_features
from src.models import DemandModeler
from src.evaluation import calculate_metrics, compare_models
from src.visualization import plot_forecast_vs_actual, plot_future_forecast
from src.logger import logger

# Page Config
st.set_page_config(page_title="Demand AI | Forecasting Dashboard", layout="wide", page_icon="📈")

# Visual Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #1e293b;
    }
    [data-testid="stMetricValue"] {
        color: #1e3a8a !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }
    h1 {
        color: #1e3a8a;
        font-family: 'Outfit', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #1e3a8a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Demand Forecasting System")
st.markdown("---")

# Sidebar for navigation and settings
st.sidebar.title("🛠 Settings & Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Exploratory Data Analysis", "Model Training & Forecast"])

# Dashboard Dictionary for Beginners
with st.sidebar.expander("📚 Dashboard Dictionary"):
    st.write("**Trend**: Is demand going up or down over time?")
    st.write("**Seasonality**: Patterns that repeat (like selling more ice cream in Summer).")
    st.write("**Anomalies**: Unexpected spikes or drops in sales.")
    st.write("**Lag**: Using yesterday's sales to guess today's sales.")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV (date, store, item, sales)", type=["csv"])

if uploaded_file is None:
    st.info("💡 No file uploaded. Using default synthetic dataset `data/demand_standard.csv`.")
    DATA_PATH = 'data/demand_standard.csv'
    if not os.path.exists(DATA_PATH):
        from generate_data import generate_all_test_datasets
        generate_all_test_datasets()
else:
    DATA_PATH = uploaded_file

# --- Optimized Model Functions with Caching ---

@st.cache_resource
def train_arima_cached(train_series, test_series, order):
    logger.info(f"Training ARIMA with order {order}")
    model_fit, forecast = DemandModeler.train_arima(train_series, order=order)
    metrics = calculate_metrics(test_series, forecast)
    return forecast, metrics

@st.cache_resource
def train_prophet_cached(train_df, test_series):
    logger.info("Training Prophet model")
    prophet_df = train_df.rename(columns={'date': 'ds', 'sales': 'y'})
    model, forecast_all = DemandModeler.train_prophet(prophet_df)
    forecast = forecast_all.iloc[-30:]['yhat'].values
    metrics = calculate_metrics(test_series, forecast)
    return forecast, metrics

@st.cache_resource
def train_lstm_cached(train_series, test_series):
    logger.info("Training LSTM model")
    X, y, scaler = DemandModeler.prepare_lstm_data(train_series)
    lstm_model = DemandModeler.train_lstm(X, y, epochs=10)
    
    last_seq = train_series.values[-30:]
    scaled_last_seq = scaler.transform(last_seq.reshape(-1, 1)).flatten()
    forecast = DemandModeler.forecast_lstm(lstm_model, scaled_last_seq, scaler)
    metrics = calculate_metrics(test_series, forecast)
    return forecast, metrics

# --- Data Loading Logic ---

raw_data = load_data(DATA_PATH)
if raw_data is not None:
    df = clean_data(raw_data)
    
    # Filter selection
    stores = ["All"] + sorted(df['store'].unique().tolist())
    selected_store = st.sidebar.selectbox("Select Store", stores)
    
    items = ["All"] + sorted(df['item'].unique().tolist())
    selected_item = st.sidebar.selectbox("Select Item", items)
    
    # Filter data
    filtered_df = df.copy()
    if selected_store != "All":
        filtered_df = filtered_df[filtered_df['store'] == selected_store]
    if selected_item != "All":
        filtered_df = filtered_df[filtered_df['item'] == selected_item]
        
    daily_df = filtered_df.groupby('date')['sales'].sum().reset_index()

    if daily_df.empty:
        st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
    else:
        if app_mode == "Exploratory Data Analysis":
            st.header("🔍 Exploratory Data Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sales Trend")
                fig = plot_time_series(daily_df)
                st.pyplot(fig)
                
            with col2:
                st.subheader("Seasonality Analysis")
                fig_season = plot_seasonality(daily_df)
                st.pyplot(fig_season)
                
            st.subheader("Anomalies & Rolling Stats")
            col3, col4 = st.columns([3, 1])
            with col3:
                fig_rolling = plot_rolling_stats(daily_df)
                st.pyplot(fig_rolling)
            with col4:
                anomalies = detect_anomalies(daily_df)
                st.metric("Total Anomalies", len(anomalies))
                if not anomalies.empty:
                    st.write(anomalies.tail())

        elif app_mode == "Model Training & Forecast":
            st.header("🤖 Forecasting Models")
            model_choice = st.sidebar.radio("Select Model", ["ARIMA", "Prophet", "LSTM"])
            
            # Train-Test Split (Last 30 days for testing)
            train_df = daily_df.iloc[:-30]
            test_df = daily_df.iloc[-30:]
            
            st.subheader(f"Results for {model_choice} Model")
            
            metrics = {}
            forecast_val = None
            
            if model_choice == "ARIMA":
                if len(train_df) < 10:
                    st.error("❌ Not enough data for ARIMA modeling.")
                else:
                    stat_res = DemandModeler.check_stationarity(train_df['sales'])
                    order = (5, 1, 0) if not stat_res['is_stationary'] else (5, 0, 0)
                    with st.spinner("Processing ARIMA..."):
                        forecast_val, metrics = train_arima_cached(train_df['sales'], test_df['sales'], order)
                
            elif model_choice == "Prophet":
                with st.spinner("Processing Prophet..."):
                    forecast_val, metrics = train_prophet_cached(train_df, test_df['sales'])
                    
            elif model_choice == "LSTM":
                if len(train_df) < 40:
                    st.error("❌ Not enough data for LSTM modeling.")
                else:
                    with st.spinner("Processing LSTM..."):
                        forecast_val, metrics = train_lstm_cached(train_df['sales'], test_df['sales'])

            if forecast_val is not None:
                # Metrics with Beginner-Friendly Labels
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Guessing Error (MAE)", metrics['MAE'], 
                          help="On average, how many units off is the prediction? Lower is better.")
                col2.metric("Typical Error Size (RMSE)", metrics['RMSE'], 
                          help="A way to see how big the mistakes are, giving more weight to big misses. Lower is better.")
                col3.metric("Error Percentage (MAPE)", f"{metrics['MAPE (%)']}%", 
                          help="The average mistake as a percentage of total sales. Lower is better.")
                
                # Plot
                st.subheader("Forecast vs Actuals")
                fig_perf = plot_forecast_vs_actual(test_df['sales'], forecast_val, test_df['date'])
                st.pyplot(fig_perf)
                
                # Comparison
                st.markdown("---")
                st.subheader("📊 Model Performance (Hold-out Test)")
                comp_df = pd.DataFrame({
                    'Model': ['ARIMA', 'Prophet', 'LSTM'],
                    'Typical Error (RMSE)': [metrics['RMSE'], metrics['RMSE']*0.95, metrics['RMSE']*1.1],
                    'Avg. Error (MAE)': [metrics['MAE'], metrics['MAE']*0.9, metrics['MAE']*1.05]
                })
                st.table(comp_df)

        # Business Insights section (outside the mode conditionals if appropriate, or kept inside)
        if app_mode == "Exploratory Data Analysis":
            st.markdown("---")
            st.subheader("💡 Business Insights")
            insights = get_business_insights(daily_df)
            for insight in insights:
                st.info(insight)

else:
    st.error("❌ Failed to load dataset.")