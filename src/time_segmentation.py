import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time features to the dataframe. Day type and time of day (morning etc).
    Args:
        df: The dataframe to add time features to.
    Returns:
        pd.DataFrame: The dataframe with time features added.
    """
    # add columns
    df['hour'] = df.index.hour
    df['day_type'] = df.index.dayofweek.map(lambda x: 'weekday' if x < 5 else 'weekend') # 0 is Monday, 6 is Sunday
    df['time_period'] = df['hour'].map(get_time_period)

    return df


def get_time_period(hour: int) -> str:
    """
    Get the time period of the day.
    Args:
        hour: The hour of the day.
    Returns:
        str: The time period of the day.
    """
    if hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'