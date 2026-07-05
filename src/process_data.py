import pandas as pd
import numpy as np

TIMESTAMP_COL = 'Timestamp'
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
MAX_INTERP_GAP   = 500  # 10 seconds (500 samples)
MAX_FLAG = 1000 # Based om observed data not IEEE spec, FLAG is a bitwise FLAG rather than a latency value, however 1000 should be suitable


def process_data(df: pd.DataFrame, max_gap: int = MAX_INTERP_GAP) -> tuple[pd.DataFrame, pd.Series]:
    """
    Processes a single data frame
    1. Removes duplicates (none found in observed data, included step for safety)
    2. Finds missing indexs and interpolates (done 2nd as requires no NaN data)
    3. Finds, flags or interpolate gps gaps
    4. Drops unfillable NaN gaps
    5. Applies segment ID's to avoid gap bridging
    Currently assumes index problem gaps will never exceed max interpolation gap
    Returns:
     Tuple of updated dataframe and series of booleans showing where data was interpolated
    TODO: Add randomness based on the datas variance when interpolating?
    """
    # 1. Remove duplicates
    df = df.loc[~df.index.duplicated(keep='first')]

    # 2. Find and interpolate missing index
    ideal_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='20ms')
    df = df.reindex(ideal_index)

    print(f"  NaN rows after reindex: {df[FREQ_COL].isna().sum()}")
    was_missing = df[FREQ_COL].isna()

    df = df.interpolate(method='linear', limit=max_gap)

    # 3. Find and flag or interpolate gps gaps
    df = find_flag_gps(df, max_gap)

    still_nan = df[FREQ_COL].isna()
    interpolated_mask = was_missing & ~still_nan

    print(f"  Interpolated rows: {interpolated_mask.sum()}")
    print(f"  NaN rows to drop: {still_nan.sum()}")

    # 4. Drop unfillable NaN rows
    df = df.dropna(subset=DATA_COLS)

    # 5. Assign segment IDs so sliding windows never straddle a gap
    dt = df.index.to_series().diff()
    expected = pd.Timedelta(f'{SAMPLE_PERIOD_MS}ms')
    df['segment_id'] = (dt > expected * 1.5).cumsum()

    print(f"  Final shape: {df.shape}, Segments: {df['segment_id'].nunique()}")

    return df, interpolated_mask


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
            # Start of a bad gap
            start = (i, ts)
            in_bad_gps = True
        elif value < MAX_FLAG and in_bad_gps:
            # end of a bad gap
            in_bad_gps = False
            if i - start[0] >= max_gap:
                # gap too big, flag it
                _flag_data(df, start[1], df.index[i - 1])
            else:
                # interpolate the gap
                df = interpolate_gap(df, start[1], df.index[i - 1], max_gap)

    # Run extends to the last row
    if in_bad_gps:
        last_ts = df.index[-1] + pd.Timedelta(f"{SAMPLE_PERIOD_MS}ms")
        if len(df) - start[0] >= max_gap:
            _flag_data(df, start[1], last_ts)
        else:
            df = interpolate_gap(df, start[1], last_ts, max_gap)

    return df


def _flag_data(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """
    Sets all data from [start, end] to NaN inplace, inclusive of both start and end
    """
    mask = (df.index >= start) & (df.index <= end)
    df.loc[mask, DATA_COLS] = np.nan


def interpolate_gap(df: pd.DataFrame, start : pd.Timestamp, end : pd.Timestamp, max_gap: int) -> pd.DataFrame:
    """
    Given a start and end index of a region in the data, this function overwrites
    all data columns where necessary using linear interpolation, returns modified df.
    """
    df = df.copy()
    df.loc[start:end, DATA_COLS] = np.nan
    len_gap = df.index.get_loc(end) - df.index.get_loc(start)
    if len_gap >= max_gap:
        return df
    df.loc[:, DATA_COLS] = df[DATA_COLS].interpolate(method="linear", limit=max_gap)
    return df