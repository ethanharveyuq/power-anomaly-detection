import pandas as pd
import numpy as np
from scipy import stats

# Possible additions
# - ROC curve analysis
# - Isolation forest analysis
# - T score analysis

def find_all_anomalies(df: pd.DataFrame, column: str = 'Voltage') -> pd.DataFrame:
    """
    Find all anomalies in the given dataframe.
    Args:
        df: The dataframe to find anomalies in.
        column: The column to find anomalies in.
    Returns:
        pd.DataFrame: The dataframe with the anomalies added.
    """
    # Calculate the rolling window anomaly
    df = rolling_window_anomaly(df, column=column)
    # Calculate the same hour anomaly
    df = same_hour_anomaly(df, column=column)
    # Calculate the combined anomaly
    df = combined_anomaly(df, column=column)
    return df

def z_score(x: float, mean: float, std: float) -> float:
    """
    Calculate the z-score for the given value.
    Args:
        x: The value to calculate the z-score for.
        mean: The mean of the data.
        std: The standard deviation of the data.
    Returns:
        float: The z-score for the given value.
    """
    return np.where(
        std > 0,
        (x - mean) / std,
        0
    ) # Avoid division by zero

def rolling_window_anomaly(df: pd.DataFrame, window: int = 120, threshold: float = 3.0, column: str = 'Voltage') -> pd.DataFrame:
    """
    Calculate the rolling window z-score for the given dataframe.
    Args:
        df: The dataframe to calculate the rolling window z-score for.
        window: The window size for the rolling window.
        threshold: The threshold for the z-score.
        column: The column to calculate the rolling window z-score for.
    Returns:
        pd.DataFrame: The dataframe with the rolling window z-score added.
    """
    df['rolling_mean'] = df[column].rolling(window=window, center=True).mean() # Change center to False if going for real world accuracy (oly past is known)
    df['rolling_std'] = df[column].rolling(window=window, center=True).std()
    df['z_score'] = z_score(df[column], df['rolling_mean'], df['rolling_std'])
    df['anomaly_rolling'] = df['z_score'].abs() > threshold
    return df

def same_hour_anomaly(df: pd.DataFrame, threshold: float = 3.0, column: str = 'Voltage') -> pd.DataFrame:
    """
    Calculate the same hour anomaly for the given dataframe. 
    Checks if the current value is significantly different from the mean of the same hour in different days.
    Args:
        df: The dataframe to calculate the same hour anomaly for.
        threshold: The threshold for the z-score.
        column: The column to calculate the same hour anomaly for.
    Returns:
        pd.DataFrame: The dataframe with the same hour anomaly added.
    """
    # Create a dataframe with the mean and std of the same hour in different days
    hourly_stats = df.groupby(['day_type', 'hour'])[column].agg(['mean', 'std'])
    df['expected_mean'] = df.set_index(['day_type', 'hour']).index.map(hourly_stats['mean'])
    df['expected_std'] = df.set_index(['day_type', 'hour']).index.map(hourly_stats['std'])
    df['expected_z_score'] = z_score(df[column], df['expected_mean'], df['expected_std'])
    df['anomaly_same_hour'] = df['expected_z_score'].abs() > threshold
    return df

def combined_anomaly(df: pd.DataFrame, column: str = 'Voltage') -> pd.DataFrame:
    """
    Determines whether the data is anomalous based on the rolling window and same hour anomalies.
    Args:
        df: The dataframe to determine the combined anomaly for.
        column: The column to determine the combined anomaly for.
    Returns:
        pd.DataFrame: The dataframe with the combined anomaly added.
    """
    df['anomaly_combined'] = df['anomaly_rolling'] & df['anomaly_same_hour'] # Both must be true for an anomaly
    return df