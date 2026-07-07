"""
Path between dataset and pytorch 
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from .PMUData import PMUData

class PMUDataset(Dataset):
    def __init__(self, data: PMUData) -> None:
        """
        """
        self.data = data
        self.feature_df = data.feature_df
        self.labels_df = data.labels_df


    def __len__(self) -> int:
        """
        Returns length of dataset (number of windows)
        """
        return len(self.data.all_IDs)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the window and label pair at the index
        """
        window_id = self.data.all_IDs[idx] # should be the same
        window_df = self.feature_df.loc[window_id]
        window = window_df.to_numpy(dtype=np.float32)
        window = torch.tensor(window)
        label = self.labels_df.loc[window_id]
        label = torch.tensor(label)
        return window, label

