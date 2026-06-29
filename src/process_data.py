import pandas as pd
import numpy as np

TIMESTAMP_COL = "Timestamp"
FREQ_COL      = "FREQ"
FLAG_COL      = "FLAG"

VOLTAGE_COLS  = [
    "UA_MAG", "UA_ANG",
    "UB_MAG", "UB_ANG",
    "UC_MAG", "UC_ANG",
]

SECONDARY_VOLTAGE_COLS = [
    "UR_MAG", "UR_ANG",
    "US_MAG", "US_ANG",
    "UT_MAG", "UT_ANG",
]
CURRENT_COLS  = [
    "IA_MAG", "IA_ANG",
    "IB_MAG", "IB_ANG",
    "IC_MAG", "IC_ANG",
    "IN_MAG", "IN_ANG"
]

DATA_COLS = VOLTAGE_COLS + SECONDARY_VOLTAGE_COLS + CURRENT_COLS + [FREQ_COL]

SAMPLE_PERIOD_MS = 20
MAX_INTERP_GAP   = 10 * 1000/SAMPLE_PERIOD_MS  # 10 seconds
MAX_FLAG = 1000


def process_data(df: pd.DataFrame, max_gap: int = MAX_INTERP_GAP) -> tuple[pd.DataFrame, pd.Series]:
    """
    Processes a single data frame
      1. Remove duplicates (keep best fit)
      2. Reindex to uniform 20ms grid
      3. Linear interpolate gaps ≤ max_gap samples; flag larger gaps (GPS)
      4. Drop boundary NaNs
    Currently assumes index problem gaps will never exceed max interpolation gap
    Returns:
     Tuple of updated dataframe and series of booleans showing where data was interpolated
    """

    # 1. Remove Duplicates

def find_gaps(df : pd.DataFrame) -> pd.DataFrame:
    """
    Finds (and flags?) gps gaps in the dataframe
    """
    df = df.copy()
    df.loc[start:end, DATA_COLS] = np.nan
    return df


def interpolate(start : pd.Timestamp, end : pd.Timestamp) -> pd.DataFrame:
    """
    Given a start and end index of a region in the data, this function overwrites
    all data columns where necessary using linear interpolation, returns modified df.
    """
    df = df.copy()
    df.loc[start:end, DATA_COLS] = df.loc[start:end, DATA_COLS].interpolate(method="linear")
    return df
    

def flag_gaps(df : pd.DataFrame, start : pd.Timestamp, end : pd.Timestamp) -> pd.DataFrame:
    """
    Sets all data columns to NaN between start and end timestamps (inclusive).
    Used to mark GPS loss windows so they survive reindexing but are never
    interpolated or trained on. Returns modified df (not inplace).
    """
    df = df.copy()
    df.loc[start:end, DATA_COLS] = np.nan
    return df

def get_best_fit(df : pd.DataFrame, location : pd.Timestamp) -> pd.Dataframe:
    """
    Given the dataframe and a timestamp that has duplicates, removes the duplicate/s with
    the worse data fit with adjacent values
    """
