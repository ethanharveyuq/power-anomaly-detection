import pandas as pd
import numpy as np

def inject_attacks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject attacks into the data. Four attacks are injected: random spike attacks, gradual drift attacks,
    sensor outage attacks, and noise attacks.
    Args:
        df: The dataframe to inject attacks into.
    Returns:
        pd.DataFrame: The dataframe with attacks injected.
    """
    # RANDOM SPIKE ATTACKS
    n = len(df)
    # for voltage column
    spike_indices = np.random.choice(n, size=n//10000, replace=False)
    df.iloc[spike_indices, df.columns.get_loc('Voltage')] = df.iloc[spike_indices, df.columns.get_loc('Voltage')] * 1.1 # double the voltage

    # for global active power column
    spike_indices = np.random.choice(n, size=n//10000, replace=False)
    df.iloc[spike_indices, df.columns.get_loc('Global_active_power')] = df.iloc[spike_indices, df.columns.get_loc('Global_active_power')] * 1.1 # double the global active power

    # GRADUAL DRIFT ATTACKS
    # for global intensity column
    drift_indice = np.random.choice(n)
    drift_length = np.random.randint(100, 1000)
    drift_values = np.linspace(0, 20, drift_length) # 20 is the maximum drift value
    df.iloc[drift_indice:drift_indice+drift_length, df.columns.get_loc('Global_intensity')] += drift_values

    # SENSOR OUTAGE ATTACKS
    # for voltage column
    outage_indices = np.random.choice(n, size=5, replace=False) # 5 is the number of outages
    for indice in outage_indices:
        outage_length = np.random.randint(100, 1000)
        df.iloc[indice:indice+outage_length, df.columns.get_loc('Voltage')] = df.iloc[indice, df.columns.get_loc('Voltage')] # Constant value at the outage indice

    # NOISE ATTACKS
    # for global active power column
    noise_indices = np.random.choice(n, size=n//100, replace=False)
    for indice in noise_indices:
        noise_length = np.random.randint(100, 1000)
        actual_length = min(noise_length, len(df) - indice)  # cap at end of dataframe
        noise_values = np.random.normal(0, 2, actual_length)  # 2 is the standard deviation
        df.iloc[indice:indice+actual_length, df.columns.get_loc('Global_active_power')] += noise_values

    return df
