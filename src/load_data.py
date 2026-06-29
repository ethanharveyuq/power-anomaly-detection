import pandas as pd

def load_data() -> pd.DataFrame:
    """
    Load the data from the csv file.
    Returns:
        pd.DataFrame: The loaded data.
    """
    # load data
    df = pd.read_csv('../data/Bd18850_2021100100.csv', low_memory=False)

    # Formatting and renaming columns
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S.%f')
    new_names = {'[10772] NOJA_AUSNET!BD18850:FREQ' : 'Frequency',
            '[10773] NOJA_AUSNET!BD18850:DFDT' : 'DF/DT',
            '[10774] NOJA_AUSNET!BD18850:FLAG' : 'GPS_Flag',
            '[10775] NOJA_AUSNET!BD18850-UA:MAG' : 'VA_Mag',
            '[10776] NOJA_AUSNET!BD18850-UA:ANG' : 'VA_Ang',
            '[10777] NOJA_AUSNET!BD18850-UB:MAG' : 'VB_Mag',
            '[10778] NOJA_AUSNET!BD18850-UB:ANG' : 'VB_Ang',
            '[10779] NOJA_AUSNET!BD18850-UC:MAG' : 'VC_Mag',
            '[10780] NOJA_AUSNET!BD18850-UC:ANG' : 'VC_Ang',
            '[10781] NOJA_AUSNET!BD18850-UR:MAG' : 'VR_Mag',
            '[10782] NOJA_AUSNET!BD18850-UR:ANG' : 'VR_Ang',
            '[10783] NOJA_AUSNET!BD18850-US:MAG' : 'VS_Mag',
            '[10784] NOJA_AUSNET!BD18850-US:ANG' : 'VS_Ang',
            '[10785] NOJA_AUSNET!BD18850-UT:MAG' : 'VT_Mag',
            '[10786] NOJA_AUSNET!BD18850-UT:ANG' : 'VT_Ang',
            '[10787] NOJA_AUSNET!BD18850-IA:MAG' : 'IA_Mag',
            '[10788] NOJA_AUSNET!BD18850-IA:ANG' : 'IA_Ang',
            '[10790] NOJA_AUSNET!BD18850-IB:MAG' : 'IB_Mag',
            '[10791] NOJA_AUSNET!BD18850-IB:ANG' : 'IB_Ang',
            '[10792] NOJA_AUSNET!BD18850-IC:MAG' : 'IC_Mag',
            '[10793] NOJA_AUSNET!BD18850-IC:ANG' : 'IC_Ang',
            '[10794] NOJA_AUSNET!BD18850-IN:MAG' : 'IN_Mag',
            '[10795] NOJA_AUSNET!BD18850-IN:ANG' : 'IN_Ang'
            }

    df.rename(columns=new_names, inplace=True)


    df.set_index('Timestamp', inplace=True)
    df.drop(columns=['[10754] NOJA_AUSNET!BD18850:ALOG1', '[10755] NOJA_AUSNET!BD18850:ALOG2'], inplace=True)

    return df