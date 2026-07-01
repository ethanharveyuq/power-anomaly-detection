import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the data from the csv file.
    Returns:
        pd.DataFrame: The loaded data.
    """
    # load data
    df = pd.read_csv(filepath, low_memory=False)

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