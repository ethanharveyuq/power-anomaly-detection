import pandas as pd
import json

def generate_anomaly_summary(df: pd.DataFrame, output_file_path: str = 'output.json') -> dict:
    """
    Generate a summary of the anomalies in the given dataframe.
    Args:
        df: The dataframe to generate the summary for.
    Returns:
        dict: The summary of the anomalies.
    """
    anomaly_breakdown = {
        'rolling_window': int(df['anomaly_rolling'].sum()),
        'same_hour': int(df['anomaly_same_hour'].sum()),
        'correlation': int(df['correlation_anomaly'].sum()),
        'combined': int(df['anomaly_combined'].sum()),
    }

    # Find biggest/longest anomalies
    condition = (df['anomaly_combined'] == True) | (df['correlation_anomaly'] == True)
    df['group_id'] = (condition != condition.shift()).cumsum()
    grouped = df.groupby('group_id')
    longest_anomalies = grouped.size().sort_values(ascending=False).head(10) # top 10 longest anomalies, need to add specific columns
    anomaly_breakdown['longest_anomalies'] = longest_anomalies.to_dict()

    anomaly_times = grouped['Global_active_power'].agg(
    start=lambda x: x.index.min(),
    end=lambda x: x.index.max(),
    duration=lambda x: len(x),
    mean_value='mean'
    )

    with open(output_file_path, 'w') as json_file: # Save to json file, overwrites/creates new file
        json.dump(anomaly_breakdown, json_file, indent=4)
    return anomaly_breakdown