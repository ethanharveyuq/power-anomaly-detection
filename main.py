from src.datasets.PMUData import PMUData
from src.datasets.dataset import PMUDataset
from src.models.gpt4ts import gpt4ts
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import re


"""
config = {

    "train data": "...",

    "validation data": "...",

    "test data": "...",

    "window length": 500,

    "stride": 250,

    "columns": ["FREQ"],

    "batch size": 32,

    "epochs": 20,

    "learning rate": 1e-4,

    "seed": 42,

    "gpu": 0,

    "patch_size": ...,

    "d_model": ...,

    "dropout": ...

    "test": "test only"
}
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
    
def run(config):
    # Step 2: Create seeds

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Step 3: Select device (CPU, GPU)
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')

    # Step 4: Load Data
    train_data = PMUData(config['train data'], config)
    validation_data = PMUData(config['acceptance data'], config)
    test_data = PMUData(config['test data'], config)

    # Step 6: create PMUDataset object wrappers
    train_dataset = PMUDataset(train_data)
    validation_dataset = PMUDataset(validation_data)
    test_dataset = PMUDataset(test_data)

    # Step 7: Create Dataloaders (create mini batches)
    train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=32,      # Group data into chunks of 32
    shuffle=True,       # Mix up data order every epoch
    num_workers=2,      # Use 2 CPU subprocesses to load data parallelly
    pin_memory=True     # Speed up data copy to GPU memory
    )

    # Step 8: Create GPT4TS model
    model = gpt4ts(config, train_data)
    model.to(device)

    # Step 9: optimiser
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=0.01
    )

    # Step 10: Loss
    criterion = nn.CrossEntropyLoss() # for classification

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

if __name__ == "__main__":
    run()
