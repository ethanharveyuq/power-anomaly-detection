import pandas as pd
import numpy as np

# TO ADD
# - Linear regression analysis
# - Pairwise Z score analysis

def correlation_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check the correlations of the given dataframe.
    Args:
        df: The dataframe to check the correlations of.
    Returns:
        pd.DataFrame: The dataframe with the correlations added.
    """
    df = rolling_window_correlation(df)
    df = ratio_anomaly(df)

    df['correlation_anomaly'] = df['anomaly_rolling'] & (df['anomaly_ratio'] | df['anomaly_unmetered'])
    return df

def rolling_window_correlation(df: pd.DataFrame, window: int = 120, threshold: float = 0.5, column1: str = 'Global_intensity', column2: str = 'Global_active_power') -> pd.DataFrame:
    """
    Calculate the rolling window correlation for the given dataframe.
    Args:
        df: The dataframe to calculate the rolling window correlation for.
        window: The window size for the rolling window.
        column1: The first column to calculate the rolling window correlation for.
        column2: The second column to calculate the rolling window correlation for.
    Returns:
        pd.DataFrame: The dataframe with the rolling window correlation added.
    """
    df['rolling_correlation'] = df[column1].rolling(window=window, center=True).corr(df[column2]) # Change center to False if going for real world accuracy (oly past is known)
    df['anomaly_rolling'] = df['rolling_correlation'].abs() > threshold
    return df

# Adjust when comparing 2 columns
def ratio_anomaly(df: pd.DataFrame, column1: str = 'Global_active_power', column2: str = 'Sub_metering_1', column3: str = 'Sub_metering_2', column4: str = 'Sub_metering_3') -> pd.DataFrame:
    """
    Calculate the ratio anomaly for the given dataframe.
    Args:
        df: The dataframe to calculate the ratio anomaly for.
        threshold: The threshold for the ratio.
        column1: The first column to calculate the ratio anomaly for.
        column2: The second column to calculate the ratio anomaly for.
        column3: The third column to calculate the ratio anomaly for.
        column4: The fourth column to calculate the ratio anomaly for.
    Returns:
        pd.DataFrame: The dataframe with the ratio anomaly added.
    """
    df['ratio'] = ((df[column2] + df[column3] + df[column4])/1000) / df[column1] # Divide by 1000 to get the ratio in kWh, should be below 1
    df['anomaly_ratio'] = df['ratio'].abs() > 1 # Should always be below 1, extend as needed
    df['unmetered'] = df[column1] - ((df[column2] + df[column3] + df[column4])/1000) # The difference between the measured and unmetered energy
    df['anomaly_unmetered'] = df['unmetered'].abs() == 0 # 0 is suspicious, adjust/expand as needed
    return df