import string
import pandas as pd

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

SAMPLE_PERIOD_MS = 20
MAX_INTERP_GAP   = 10 * 1000/SAMPLE_PERIOD_MS  # 10 seconds



def process_data(df: pd.DataFrame, max_gap: int = MAX_INTERP_GAP) -> tuple[pd.DataFrame, pd.Series]:
    """
    Processes a single data frame
    Parse timestamps
      2. Remove duplicates (keep best fit)
      3. Reindex to uniform 20ms grid
      4. Linear interpolate gaps ≤ max_gap samples; flag larger gaps
      5. Drop boundary NaNs
    """

def interpolate(start : pd.DatetimeIndex, end : pd.DatetimeIndex) -> None:
    """
    Given a start and end index of a region in the data, this function overwrites
    all data columns where necessary using linear interpolation, doing so inplace, thus returns None.
    """
    def interpolate_data(column : string):
        """
        Interpolates single data type
        """
        return


def flag_bad(start : pd.DatetimeIndex, end : pd.DatetimeIndex) -> None:
    """
    Given a start and end flags the data as bad (NaN) and thus will be avoided in processing.
    """
    return
