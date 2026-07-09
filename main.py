from src.datasets.PMUData import PMUData
from src.datasets.dataset import PMUDataset
from src.models.gpt4ts import gpt4ts
from config import parse_args, create_config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import re

    
def run(config):
    """
    """
    # Create seeds

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Select device (CPU, GPU) TODO add cuda when operating on PC
    device = torch.device('cpu')

    # Load Data
    train_data = PMUData(config['train data'], config['train pattern'], config)
    #validation_data = PMUData(config['validation data'], config['validation pattern'], config)
    #test_data = PMUData(config['test data'], config['test pattern'], config)

    # create PMUDataset object wrappers
    train_dataset = PMUDataset(train_data)
    #validation_dataset = PMUDataset(validation_data)
    #test_dataset = PMUDataset(test_data)

    # Create Dataloaders (create mini batches)
    train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=config['batch size'],      # Group data into chunks of 32
    shuffle=True,       # Mix up data order every epoch
    num_workers=2,      # Use 2 CPU subprocesses to load data parallelly
    pin_memory=True     # Speed up data copy to GPU memory
    )

    #validation_loader = DataLoader(
    #dataset=validation_dataset, 
    #batch_size=config['batch size'],      
    #shuffle=True,       
    #num_workers=2,      
    #pin_memory=True     
    #)

    #test_loader = DataLoader(
    #dataset=test_dataset, 
    #batch_size=config['batch size'],      
    #shuffle=True,       
    #num_workers=2,      
    #pin_memory=True
    #)

    # Create GPT4TS model
    model = gpt4ts(config, train_data)
    model.to(device)

    # Step 9: optimiser
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning rate"],
    weight_decay=0.01
    )

    # Step 10: Loss
    criterion = nn.CrossEntropyLoss() # for classification

    """
    # Step 11: Training loop

    best_f1 = 0.0
    for epoch in range(config["epochs"]):

        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        model.train()

        running_loss = 0.0

        for windows, labels in train_loader:

            # Move tensors onto CPU/GPU
            windows = windows.to(device)
            labels = labels.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass through GPT4TS
            outputs = model(windows)

            # Compute loss
            loss = criterion(outputs, labels)

            # Compute gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Statistics
            running_loss += loss.item()

        average_train_loss = running_loss / len(train_loader)

        print(f"Training Loss = {average_train_loss:.4f}")

        model.eval()

        validation_loss = 0.0

        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for windows, labels in validation_loader:

                windows = windows.to(device)
                labels = labels.to(device)

                outputs = model(windows)

                loss = criterion(outputs, labels)

                validation_loss += loss.item()

                # Predicted class
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        average_validation_loss = validation_loss / len(validation_loader)

        # Calculate metrics
        accuracy = ...
        precision = ...
        recall = ...
        f1 = ...

        print(f"Validation Loss = {average_validation_loss:.4f}")
        print(f"Accuracy = {accuracy:.4f}")
        print(f"F1 Score = {f1:.4f}")

        if f1 > best_f1:

            best_f1 = f1

            torch.save(model.state_dict(), "best_model.pt")

            print("New best model saved.")


    # Reload best model
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()


    # Step 15: Final testing
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for windows, labels in test_loader:

            windows = windows.to(device)
            labels = labels.to(device)

            outputs = model(windows)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    

    # Compute final metrics

    accuracy = ...
    precision = ...
    recall = ...
    f1 = ...
    confusion = ...

    print("Final Results")
    print("---------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print(confusion)
    """

if __name__ == "__main__":
    args = parse_args()
    config = create_config(args)
    run(config)
