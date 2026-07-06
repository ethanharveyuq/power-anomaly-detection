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
        self.pmu_source = {}
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
            set_num = self.num_dataframes
        
        self.num_dataframes += 1
        new_windows = self._create_windows(df, cols)

        self.windows.extend(new_windows)
        self.labels.extend([set_num] * len(new_windows))
        
        self.pmu_source[set_num] = name

        

    def _create_windows(self, df: pd.DataFrame, cols: list[str]) -> List(torch.Tensor):
        """
        From a dataframe's data, creates the windows of data 
        """
        return

    def _get_label(self, df: pd.DataFrame) -> int:
        """
        Gets the PMU label from a dataframe
        """
        return

