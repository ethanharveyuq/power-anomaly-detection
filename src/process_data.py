from turtle import end_poly
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
      1. Find missing values -> Check if duplicates around it and uses those or interpolates
      2. Linear interpolate gaps ≤ max_gap samples; flag larger gaps (GPS)
      3. Runs through all indexs (none should be missing), if duplicates remain, delete
    Currently assumes index problem gaps will never exceed max interpolation gap
    Returns:
     Tuple of updated dataframe and series of booleans showing where data was interpolated
    """

    # 1. Remove Duplicates

def find_flag_gps(df: pd.DataFrame, max_gap: int = MAX_INTERP_GAP) -> pd.DataFrame:
    """
    Identifies contiguous runs of bad gps data.
    Any run longer than max_gap samples is flagged as NaN across DATA_COLS
    (too large to safely interpolate — left for dropna() downstream).
    Runs of length <= max_gap are left untouched so interpolate() can fill them.

    Assumes df has already been reindexed to the uniform timestamp grid,
    so missing rows show up as NaN rather than absent index entries.
    """
    df = df.copy()
    in_bad_gps = False
    start = None
    
    for i, (ts, value) in enumerate(df[FLAG_COL].items()):
        if value >= MAX_FLAG and not in_bad_gps:
            start = (i, ts)
            in_bad_gps = True
        elif value < MAX_FLAG and in_bad_gps:
            in_bad_gps = False
            if i - start[0] >= MAX_INTERP_GAP:
                _flag_data(df, start[1], ts)

    # Run extends to the last row
    if in_bad_gps:
        last_ts = df.index[-1] + pd.Timedelta(f"{SAMPLE_PERIOD_MS}ms")
        if len(df) - start[0] >= max_gap:
            _flag_data(df, start[1], last_ts)

    return df


def _flag_data(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """
    Sets all data from [start, end) to NaN inplace
    """
    mask = (df.index >= start) & (df.index < end)
    df.loc[mask, DATA_COLS] = np.nan



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