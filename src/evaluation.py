import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, and MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Avoid division by zero for MAPE
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if any(non_zero) else np.nan
    
    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE (%)': round(mape, 2)
    }

def compare_models(results_dict):
    """
    results_dict: { 'ModelName': { 'MAE': val, 'RMSE': val, ... } }
    Returns a formatted comparison table.
    """
    df = pd.DataFrame(results_dict).T
    df = df.sort_values(by='RMSE')
    return df
