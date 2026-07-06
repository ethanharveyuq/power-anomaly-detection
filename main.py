import sys
import os

from src.load_data import load_data
from src.process_data import process_data
from src.dataset import PMUDataset
from pathlib import Path

# Columns included in AI categorisation
COLS = [
    "FREQ"
]


def add_to_dataset(dataset: PMUDataset, filepath: str) -> None:
    df = load_data(filepath)
    df, mask = process_data(df)
    dataset.add_dataframe(df, COLS, filepath.name)
    

def run():

    dataset = PMUDataset(250, 125)

    directory_path = Path('data/')
    # Iterate through all files in data
    for file_path in directory_path.iterdir():

        # Add data to dataset
        if file_path.is_file():
            add_to_dataset(dataset, file_path)
    
    print(len(dataset))
    to_check = [0, 450, 1245, 2657, 3000]
    for i in to_check:
        window, label = dataset[i]
        print(window)
        print(label)
        print(window.shape)

if __name__ == "__main__":
    run()
