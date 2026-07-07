"""
Read CSVs, clean them, create windows, store metadata
"""

from multiprocessing import Pool, cpu_count
import pandas as pd
from pathlib import Path
import re
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

    was_missing = df[FREQ_COL].isna()

    df = df.interpolate(method='linear', limit=max_gap)

    # 3. Find and flag or interpolate gps gaps
    df = find_flag_gps(df, max_gap)

    still_nan = df[FREQ_COL].isna()
    interpolated_mask = was_missing & ~still_nan

    # 4. Drop unfillable NaN rows
    df = df.dropna(subset=DATA_COLS)

    # 5. Assign segment IDs so sliding windows never straddle a gap
    dt = df.index.to_series().diff()
    expected = pd.Timedelta(f'{SAMPLE_PERIOD_MS}ms')
    df['segment_id'] = (dt > expected * 1.5).cumsum()

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


# From GPT4TS
class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


# CONFIG Structure
# config = {
# 'file pattern' = A Regex pattern for which files to load, must be compile()
# 'window length' = Size of windows to use
# 'stride' = Stride length between windows
# 'columns' = columns that will be included in windows
#
#
#
# }


class PMUData(BaseData):

    def __init__(self, root_dir: str, config: dict):
        # class names
        self.class_names = []
        # columns being used in windows
        self.feature_names = self.config['columns']

        self.config = config
        self.max_seq_len = config['window length']

        # Process 
        self.process_all_files(root_dir)
        self.num_classes = len(self.class_names)
        return


    def process_all_files(self, root_dir: str) -> None:
        """
        Iterates through all files in directory, loads, processes, makes windows
        """
        
        labels = []
        feature_frames = []
        win_idx = 0
        files_loaded = 0

        directory_path = Path(root_dir)
        for file_path in sorted(directory_path.iterdir()):
            # process and add
            if file_path.is_file() and self.config['file pattern'].match(file_path.name):
                print(f"Loading {file_path}")
                # storing class name
                self.class_names.append(file_path.name[:-15]) # chop off timestamp just keep PMU id TODO change with .stem and regex
                # load df
                df = self.load_single_file(file_path)
                # clean df
                df = self.clean_dataframe(df)

                # iterate over all segments and make windows
                for _, segment_df in df.groupby("segment_id"):
                    # create the windows for that segment
                    df_windows, new_win_idx = self.generate_windows(segment_df, new_win_idx)
                    # add to list
                    feature_frames.extend(df_windows)

                # add window indexs to label then icrement
                # should line up with windows added, TODO should be moved to increment during window creation
                for _ in range(win_idx, new_win_idx):
                    labels.append(files_loaded) # the windows in that range belong to the ID

                win_idx = new_win_idx
                files_loaded += 1
        
        # Window IDs
        self.all_IDs = list(range(win_idx))
        # create the feature_df and labels_df from the feature_frames and labels lists
        self.feature_df = pd.concat(feature_frames)
        self.feature_df = self.feature_df.set_index('WindowID')
        self.labels_df = pd.DataFrame(labels, columns=['Label']) # could switch label to file or pmuid?
        self.labels_df.index.name = 'WindowID'


        
    def load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Load the data from the csv file.
        Returns:
            pd.DataFrame: The loaded data.
        """
        # load data
        df = pd.read_csv(file_path, low_memory=False)

        # Formatting and renaming columns
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S.%f')
        new_names = {'[10772] NOJA_AUSNET!BD18850:FREQ' : 'FREQ',
                '[10773] NOJA_AUSNET!BD18850:DFDT' : 'DF/DT',
                '[10774] NOJA_AUSNET!BD18850:FLAG' : 'FLAG',
                '[10775] NOJA_AUSNET!BD18850-UA:MAG' : 'UA_MAG',
                '[10776] NOJA_AUSNET!BD18850-UA:ANG' : 'UA_ANG',
                '[10777] NOJA_AUSNET!BD18850-UB:MAG' : 'UB_MAG',
                '[10778] NOJA_AUSNET!BD18850-UB:ANG' : 'UB_ANG',
                '[10779] NOJA_AUSNET!BD18850-UC:MAG' : 'UC_MAG',
                '[10780] NOJA_AUSNET!BD18850-UC:ANG' : 'UC_ANG',
                '[10781] NOJA_AUSNET!BD18850-UR:MAG' : 'UR_MAG',
                '[10782] NOJA_AUSNET!BD18850-UR:ANG' : 'UR_ANG',
                '[10783] NOJA_AUSNET!BD18850-US:MAG' : 'US_MAG',
                '[10784] NOJA_AUSNET!BD18850-US:ANG' : 'US_ANG',
                '[10785] NOJA_AUSNET!BD18850-UT:MAG' : 'UT_MAG',
                '[10786] NOJA_AUSNET!BD18850-UT:ANG' : 'UT_ANG',
                '[10787] NOJA_AUSNET!BD18850-IA:MAG' : 'IA_MAG',
                '[10788] NOJA_AUSNET!BD18850-IA:ANG' : 'IA_ANG',
                '[10790] NOJA_AUSNET!BD18850-IB:MAG' : 'IB_MAG',
                '[10791] NOJA_AUSNET!BD18850-IB:ANG' : 'IB_ANG',
                '[10792] NOJA_AUSNET!BD18850-IC:MAG' : 'IC_MAG',
                '[10793] NOJA_AUSNET!BD18850-IC:ANG' : 'IC_ANG',
                '[10794] NOJA_AUSNET!BD18850-IN:MAG' : 'IN_MAG',
                '[10795] NOJA_AUSNET!BD18850-IN:ANG' : 'IN_ANG'
                }

        df = df.rename(columns=new_names)

        df = df.set_index('Timestamp').sort_index()
        df = df.drop(columns=['[10754] NOJA_AUSNET!BD18850:ALOG1', '[10755] NOJA_AUSNET!BD18850:ALOG2'])
        print(df['FLAG'].value_counts().sort_index())

        return df

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans dataframe
        """
        df, interpolate_mask = process_data(df)
        return df
    
    def generate_windows(self, df: pd.DataFrame, win_idx: int) -> tuple(list[pd.DataFrame], int):
        """
        Converts a dataframe into a list of dataframe windows with relevant data (from config columns)
        and indexed following the win_idx passed to it
        Returns:
            A tuple containing
                1. list of dataframes with relevant columns and window index
                2. The next win_idx to use
        """
        window_length = self.config['window length']
        stride = self.config['stride']
        windows = []
    
        if len(df) < window_length:
            return [], win_idx

        start = 0
        while start + window_length <= len(df):

            # Extract window
            window_df = df.iloc[start : start + window_length]
            feature_df = window_df[self.feature_names] # TODO check if keeps relative order
            feature_df = feature_df.copy()
            feature_df["WindowID"] = win_idx
            windows.append(feature_df)
            
            # Move window
            start += stride
            win_idx += 1
        
        return windows, win_idx