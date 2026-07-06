import array
from ast import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class PMUDataset(Dataset):
    def __init__(self, n_sources: int, window_size: int, stride: int) -> None:
        """
        Initialises dataset arrays and size, window size and stride
        Includes mapping between PMU_Id and integer
        """
        return
    
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

    def add_dataframe(self, df: pd.DataFrame) -> None:
        """
        Adds an extra dataframe to the dataset, allows for gradual creation,
        avoiding high memory costs
        """
        return

    def _create_windows(self, df: pd.DataFrame) -> List(torch.Tensor):
        """
        From a dataframe's data, creates the windows of data 
        """
        return

    def _get_label(self, df: pd.DataFrame) -> int:
        """
        Gets the PMU label from a dataframe
        """
        return

