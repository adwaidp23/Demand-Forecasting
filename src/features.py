import pandas as pd
import numpy as np

def create_time_features(df):
    """Extracts date-related features."""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def create_lag_features(df, lags=[1, 7, 30]):
    """Creates lag features for the sales column."""
    df = df.copy()
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)
    return df

def create_rolling_features(df, windows=[7, 30]):
    """Creates rolling mean features."""
    df = df.copy()
    for window in windows:
        # Use shift(1) to avoid data leakage
        df[f'sales_roll_mean_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
            lambda x: x.shift(1).rolling(window=window).mean()
        )
    return df

def prepare_features(df):
    """Combines all feature engineering steps."""
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    
    # Drop rows with NaN values created by shift/rolling
    df = df.dropna().reset_index(drop=True)
    return df
