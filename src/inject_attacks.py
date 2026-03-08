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
    spike_indices = np.random.choice(n, size=n//100, replace=False)
    df.loc[spike_indices, 'Voltage'] = df.loc[spike_indices, 'Voltage'] * 2 # double the voltage

    # for global active power column
    spike_indices = np.random.choice(n, size=n//100, replace=False)
    df.loc[spike_indices, 'Global_active_power'] = df.loc[spike_indices, 'Global_active_power'] * 2 # double the global active power

    # GRADUAL DRIFT ATTACKS
    # for global intensity column
    drift_indice = np.random.choice(n)
    drift_length = np.random.randint(100, 1000)
    drift_values = np.linspace(0, 20, drift_length) # 20 is the maximum drift value
    df.loc[drift_indice:drift_indice+drift_length, 'Global_intensity'] += drift_values

    # SENSOR OUTAGE ATTACKS
    # for voltage column
    outage_indices = np.random.choice(n, size=5, replace=False) # 5 is the number of outages
    for indice in outage_indices:
        outage_length = np.random.randint(100, 1000)
        df.loc[indice:indice+outage_length, 'Voltage'] = df.loc[indice, 'Voltage'] # Constant value at the outage indice

    # NOISE ATTACKS
    # for global active power column
    noise_indices = np.random.choice(n, size=n//100, replace=False)
    for indice in noise_indices:
        noise_length = np.random.randint(100, 1000)
        noise_values = np.random.normal(0, 2, noise_length) # 2 is the standard deviation
        df.loc[indice:indice+noise_length, 'Global_active_power'] += noise_values

    return df