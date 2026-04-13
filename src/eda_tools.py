import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_time_series(df, column='sales', title='Time Series Trend'):
    """Plots the time series trend."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df[column])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True)
    return plt

def plot_seasonality(df):
    """Plots weekly and monthly seasonality."""
    df['week_day'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month_name()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Weekly
    sns.boxplot(x='week_day', y='sales', data=df, ax=axes[0], order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    axes[0].set_title('Weekly Seasonality')
    
    # Monthly
    sns.boxplot(x='month', y='sales', data=df, ax=axes[1])
    axes[1].set_title('Monthly Seasonality')
    
    plt.tight_layout()
    return fig

def plot_rolling_stats(df, window=30):
    """Plots rolling mean and standard deviation."""
    rolling_mean = df['sales'].rolling(window=window).mean()
    rolling_std = df['sales'].rolling(window=window).std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['sales'], alpha=0.5, label='Actual')
    plt.plot(df['date'], rolling_mean, label='Rolling Mean', color='red')
    plt.plot(df['date'], rolling_std, label='Rolling Std', color='black')
    plt.legend()
    plt.title(f'Rolling Mean & Std (Window={window})')
    return plt

def detect_anomalies(df, window=30, sigma=3):
    """Detects anomalies using rolling Z-score."""
    rolling_mean = df['sales'].rolling(window=window).mean()
    rolling_std = df['sales'].rolling(window=window).std()
    
    upper_bound = rolling_mean + (sigma * rolling_std)
    lower_bound = rolling_mean - (sigma * rolling_std)
    
    anomalies = df[(df['sales'] > upper_bound) | (df['sales'] < lower_bound)]
    return anomalies

def get_business_insights(df):
    """Returns 3-5 business insights."""
    insights = []
    
    # 1. Trend
    start_val = df.iloc[:30]['sales'].mean()
    end_val = df.iloc[-30:]['sales'].mean()
    trend_pct = ((end_val - start_val) / start_val) * 100
    insights.append(f"Demand has {'increased' if trend_pct > 0 else 'decreased'} by {abs(trend_pct):.2f}% over the observed period.")
    
    # 2. Peak Day
    df['day'] = df['date'].dt.day_name()
    peak_day = df.groupby('day')['sales'].mean().idxmax()
    insights.append(f"Peak sales typically occur on {peak_day}s.")
    
    # 3. Peak Month
    df['month'] = df['date'].dt.month_name()
    peak_month = df.groupby('month')['sales'].mean().idxmax()
    insights.append(f"Highest seasonal demand is usually observed in {peak_month}.")
    
    # 4. Volatility
    volatility = df['sales'].std() / df['sales'].mean()
    insights.append(f"The demand volatility (coefficient of variation) is {volatility:.2f}.")
    
    return insights
