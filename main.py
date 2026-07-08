from src.datasets.PMUData import PMUData
from src.datasets.dataset import PMUDataset
import torch

import re


"""
Config will have:
data directory
train regex
test regex
window length
stride
batch size
learning rate
epochs
columns
GPT4TS hyperparameters
"""

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

    # Step 1: create config from args

    config = {
        'file pattern' : PATTERN,
        'window length' : WINDOW_LEN,
        'stride' : STRIDE,
        'columns' : COLS
        }

    # Step 2: Create seeds



    # Step 3: Select device (CPU, GPU)


    # Step 4: Load Training data

    # Step 4.5: (maybe) Load Acceptance data

    # Step 5: Load testing data

    # Step 6: create PMUDataset object wrappers


    # Step 7: Create Dataloaders (create mini batches)

    # Step 8: Create GPT4TS model


    # Step 9: optimiser

    # Step 10: Loss


    # Step 11: Training loop
    # for every epoch:
    #   for every batch:
    #       window batch, label batch
    #       Move tensors to GPU
    #
    #            ↓
    #
    #            Forward pass
    #
    #            ↓
    #
    #            Compute loss
    #
    #            ↓
    #
    #            Backpropagation
    #
    #            ↓
    #
    #            Optimizer step
    #
    #           ↓
    #
    #            Zero gradients
    #


    # Step 12: Validation, compute f1 loss accuracy confusion matrix



    # Step 13: If accuracy increased, save model


    # Step 14: Reload    


    # Step 15: Final testing



    



    data = PMUData('data/', config)

if __name__ == "__main__":
    run()
