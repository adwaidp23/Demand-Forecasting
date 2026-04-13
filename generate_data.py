import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_all_test_datasets():
    """Generates multiple datasets for different test scenarios."""
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
        
    scenarios = {
        'demand_standard.csv': {'noise': 5, 'trend_slope': 20, 'missing': 0},
        'demand_high_volatility.csv': {'noise': 20, 'trend_slope': 10, 'missing': 0},
        'demand_strong_trend.csv': {'noise': 5, 'trend_slope': 100, 'missing': 0},
        'demand_missing_values.csv': {'noise': 5, 'trend_slope': 20, 'missing': 0.1}
    }
    
    for filename, params in scenarios.items():
        df = generate_demand_data(
            num_days=730, 
            noise_scale=params['noise'], 
            trend_end=params['trend_slope'],
            missing_prob=params['missing']
        )
        df.to_csv(f'data/{filename}', index=False)
        print(f"Generated {filename}")

def generate_demand_data(num_days=730, num_stores=2, num_items=3, noise_scale=5, trend_end=20, missing_prob=0):
    """
    Generates a synthetic demand dataset with specific characteristics.
    """
    np.random.seed(42)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    data = []
    for store in range(1, num_stores + 1):
        for item in range(1, num_items + 1):
            base_sales = np.random.randint(50, 100)
            trend = np.linspace(0, trend_end, num_days)
            weekly_seasonality = [10 if d.weekday() >= 5 else 0 for d in dates]
            annual_seasonality = 15 * np.sin(2 * np.pi * np.array(range(num_days)) / 365.25)
            noise = np.random.normal(0, noise_scale, num_days)
            
            sales = base_sales + trend + weekly_seasonality + annual_seasonality + noise
            sales = np.maximum(sales, 0)
            
            for i, date in enumerate(dates):
                # Introduce missing values if requested
                if np.random.random() < missing_prob:
                    val = np.nan
                else:
                    val = int(sales[i])
                    
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'store': f"Store_{store}",
                    'item': f"Item_{item}",
                    'sales': val
                })
                
    return pd.DataFrame(data)

if __name__ == "__main__":
    generate_all_test_datasets()
