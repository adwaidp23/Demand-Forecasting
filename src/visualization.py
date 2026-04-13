import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_forecast_vs_actual(y_true, y_pred, dates, title="Actual vs Predicted"):
    """Plots actual values vs predictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='Actual', color='blue', alpha=0.6)
    plt.plot(dates, y_pred, label='Predicted', color='orange', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt

def plot_future_forecast(historical_dates, historical_values, forecast_dates, forecast_values, title="Demand Forecast"):
    """Plots historical data alongside future forecast."""
    plt.figure(figsize=(12, 6))
    plt.plot(historical_dates, historical_values, label='Historical', color='blue')
    plt.plot(forecast_dates, forecast_values, label='Forecast', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt

def plot_error_distribution(y_true, y_pred):
    """Plots the distribution of residuals."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True, color='purple')
    plt.title('Residuals Distribution')
    plt.xlabel('Error')
    return plt
