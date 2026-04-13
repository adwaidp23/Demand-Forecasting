import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class DemandModeler:
    """Wrapper class for different forecasting models."""
    
    @staticmethod
    def check_stationarity(series):
        """Performs ADF test."""
        result = adfuller(series)
        return {
            'adf_stat': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }

    @staticmethod
    def train_arima(series, order=(5, 1, 0)):
        """Trains ARIMA model and forecasts next 30 days."""
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)
        return model_fit, forecast

    @staticmethod
    def train_prophet(df, country_code='US'):
        """
        Trains Prophet model.
        Expects df with columns ['ds', 'y']
        """
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.add_country_holidays(country_name=country_code)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        return model, forecast

    @staticmethod
    def prepare_lstm_data(series, n_steps=30):
        """Prepares sequences for LSTM."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - n_steps):
            X.append(scaled_data[i:(i + n_steps), 0])
            y.append(scaled_data[i + n_steps, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, scaler

    @staticmethod
    def train_lstm(X, y, epochs=20, batch_size=32):
        """Trains LSTM model."""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return model

    @staticmethod
    def forecast_lstm(model, last_sequence, scaler, n_steps=30):
        """Forecasts future values using LSTM."""
        current_seq = last_sequence.copy()
        forecasts = []
        
        for _ in range(30):
            pred = model.predict(current_seq.reshape(1, n_steps, 1), verbose=0)
            forecasts.append(pred[0, 0])
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred[0, 0]
            
        return scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
