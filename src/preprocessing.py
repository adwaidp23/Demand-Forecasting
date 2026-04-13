import pandas as pd
import numpy as np

def load_data(file_path):
    """Loads dataset from CSV."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Performs preprocessing:
    - Date conversion
    - Sorting
    - Handling missing values
    - Aggregate to daily sales if multiple entries per day
    """
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    
    # Check for duplicates and drop
    df = df.drop_duplicates()
    
    # Sort chronologically
    df = df.sort_values(by=['store', 'item', 'date'])
    
    # Handle missing values: Linear interpolation per store/item group
    df['sales'] = df.groupby(['store', 'item'])['sales'].transform(lambda x: x.interpolate(method='linear').ffill().bfill())
    
    return df

def aggregate_sales(df):
    """Aggregates sales by date if needed (e.g., total demand across all stores)."""
    daily_df = df.groupby('date')['sales'].sum().reset_index()
    return daily_df

if __name__ == "__main__":
    # Test preprocessing
    data = load_data('data/demand_data.csv')
    if data is not None:
        cleaned_data = clean_data(data)
        print("Data preprocessed successfully.")
        print(cleaned_data.head())
