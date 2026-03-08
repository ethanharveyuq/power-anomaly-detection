import pandas as pd
import matplotlib.pyplot as plt

# TO ADD
# - Group anomalies by category

def plot_anomalies(df: pd.DataFrame, column: str = 'Voltage') -> None:
    """
    Plot the anomalies in the given dataframe.
    Args:
        df: The dataframe to plot the anomalies in.
        column: The column to plot the anomalies in.
    """
    anomalies = df[df['anomaly_combined']]
    plt.plot(df.index, df[column], label='Usage')
    plt.scatter(anomalies.index, anomalies[column], color='red', label='Anomalies')
    plt.legend()
    plt.title(f'{column} - Anomalies Highlighted')
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()

    return None