import pandas as pd

def load_data() -> pd.DataFrame:
    """
    Load the data from the csv file.
    Returns:
        pd.DataFrame: The loaded data.
    """
    # load data
    df = pd.read_csv('data/household_power_consumption.txt',
                    sep=';', # separator
                    low_memory=False, # don't load the data into memory
                    na_values=['?']) # replace '?' with NaN

    # convert date and time to datetime (adjust for your format)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                    format='%d/%m/%Y %H:%M:%S')

    # set datetime as index
    df = df.set_index('datetime')

    # drop original date and time columns
    df = df.drop(columns=['Date', 'Time'])

    return df