from src.datasets.PMUData import PMUData
from src.datasets.dataset import PMUDataset
from pathlib import Path
import re

# Columns included in AI categorisation
COLS = [
    "FREQ"
]
# File pattern to include
PATTERN = re.compile(r"^.*00\.csv$")

# Window Length
WINDOW_LEN = 500 # 10 secs

# Stride Length
STRIDE = 250 # 50% overlap
    
def run():

    config = {
        'file pattern' : PATTERN,
        'window length' : WINDOW_LEN,
        'stride' : STRIDE,
        'columns' : COLS
        }

    data = PMUData('data/', config)

if __name__ == "__main__":
    run()
