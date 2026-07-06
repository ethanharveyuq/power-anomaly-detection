import array
from ast import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class PMUDataset(Dataset):
    def __init__(self, window_size: int, stride: int) -> None:
        """
        Initialises dataset arrays and size, window size and stride
        Includes mapping between PMU_Id and integer
        """
        self.num_dataframes = 0
        self.pmu_source = {} # dict between name of pmu and number allocated
        self.window_size = window_size
        self.stride = stride
        self.windows = []
        self.labels = []
    
    def __len__(self) -> int:
        """
        Returns length of dataset (number of windows)
        """
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple(torch.Tensor, str):
        """
        Returns the window and label pair at the index
        """
        return self.windows[idx], self.labels[idx]

    def add_dataframe(self, df: pd.DataFrame, cols: list[str], name: str, set_num: int = None) -> None:
        """
        Adds an extra dataframe to the dataset, allows for gradual creation,
        avoiding high memory costs
        Parameters:
        df: dataframe to add
        cols: the columns of the dataframe to add
        name: name of pmu
        set_num: the number dataset this is
        """
        if set_num is None:
            set_num = self.num_dataframes # default value is the current dataframe number
        
        # increment dataframe number
        self.num_dataframes += 1

        # iterate over all segments
        max_segment = df['segment_id'].max()
        for i in range(max_segment + 1):
            segment_df = df.loc[df['segment_id'] == i]
            # create the windows for that segment
            self._create_windows(segment_df, cols, set_num)

        # set name of pmu
        self.pmu_source[set_num] = name


    def _create_windows(self, df: pd.DataFrame, cols: list[str], label: int) -> None:
        """
        From a dataframe's data, creates the windows of data and stores them
        Cuts off end if too smal;
        """
        if len(df) < self.window_size:
            return

        start = 0
        while start + self.window_size <= len(df):

            # Extract window
            window_df = df.iloc[start : start + self.window_size]
            feature_df = window_df[cols]

            # Convert to tensor
            window_tensor = torch.tensor(
            feature_df.values,
            dtype=torch.float32
            )
            
            # Store it
            self.windows.append(window_tensor)
            self.labels.append(label)

            # Move window
            start += self.stride

        

    def _get_label(self, df: pd.DataFrame) -> int:
        """
        Gets the PMU label from a dataframe
        """
        return

