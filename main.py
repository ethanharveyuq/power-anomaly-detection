from src.datasets.PMUData import PMUData
from src.datasets.dataset import PMUDataset
from src.models.gpt4ts import gpt4ts
from config import parse_args, create_config
from collections import Counter, deque
import sklearn.metrics as skm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import random
import os
import time
import tracemalloc

# from GPT4TS
def l2_reg_loss(model):
    for name, param in model.named_parameters():
        if name == 'out_layer.weight':  # match YOUR layer's actual name
            return torch.sum(torch.square(param))

    
def run(config):
    """
    """
    # Create seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Select device (CPU, GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load Data
    print("Loading training data...")
    if config["experiment"]: 
        train_data = PMUData(config['train data'], config['train pattern'], config)
        train_dataset = PMUDataset(train_data)
        labels = train_dataset.labels_df['Label'].values
        # Subset of data
        per_class_cap = 20 # can change

        selected_indices = []
        for cls in np.unique(labels):
            cls_indices = np.where(labels == cls)[0]
            selected_indices.extend(cls_indices[:per_class_cap].tolist())

        overfit_dataset = Subset(train_dataset, selected_indices)
        overfit_loader = DataLoader(overfit_dataset, batch_size=32, shuffle=True)
    else:
        train_data = PMUData(config['train data'], config['train pattern'], config)
        print("Loading validation data...")
        validation_data = PMUData(config['validation data'], config['validation pattern'], config)
        #print("Loading testing data")
        #test_data = PMUData(config['test data'], config['test pattern'], config)

        # create PMUDataset object wrappers
        train_dataset = PMUDataset(train_data) 
        validation_dataset = PMUDataset(validation_data)
        #test_dataset = PMUDataset(test_data)
    
        #Create Dataloaders (create mini batches)
        train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config['batch size'],      # Group data into chunks of 32
        shuffle=True,       # Mix up data order every epoch
        num_workers=2,      # Use 2 CPU subprocesses to load data parallelly
        pin_memory=True     # Speed up data copy to GPU memory
        )

        # Make validation set smaller samples to increase epoch speed
        val_samples_per_class = 50 # can tune
        val_labels = validation_dataset.labels_df['Label'].values
        rng = np.random.default_rng(config['seed'])

        selected_indices = []
        for cls in np.unique(val_labels):
            cls_indices = np.where(val_labels == cls)[0]
            if len(cls_indices) > val_samples_per_class:
                cls_indices = rng.choice(cls_indices, size=val_samples_per_class, replace=False)
            selected_indices.extend(cls_indices.tolist())

        validation_dataset = Subset(validation_dataset, selected_indices)

        validation_loader = DataLoader(
        dataset=validation_dataset, 
        batch_size=config['batch size'],      
        shuffle=True,       
        num_workers=2,      
        pin_memory=True     
        )

        #test_loader = DataLoader(
        #dataset=test_dataset, 
        #batch_size=config['batch size'],      
        #shuffle=True,       
        #num_workers=2,      
        #pin_memory=True
        #)

    # Create GPT4TS model
    print("Initialising LLM Model...")
    model = gpt4ts(config, train_data)
    model.to(device)
    print(f"Patch num: {model.patch_num}")
    print(model.feat_dim * model.patch_size)

    # Initialise optimiser
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('gpt2.'):
            backbone_params.append(param)
        else:
            head_params.append(param)  # enc_embedding, ln_proj, out_layer

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config["backbone learning rate"], 'weight_decay': 0.0},
        {'params': head_params, 'lr': config["head learning rate"], 'weight_decay': 0.01}, # maybe change to 0.0
    ])
    for i, g in enumerate(optimizer.param_groups):
        print(f"group {i}: lr={g['lr']}, num_params={sum(p.numel() for p in g['params'])}")

    # Loss model
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # for classification, 0.1 prevents over confidence


    # experiment loop
    if config["experiment"]:
        print("Beginning experimental training")
        for step in range(800):
            correct, total, total_loss = 0, 0, 0.0
            for windows, labels in overfit_loader:
                windows, labels = windows.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(windows)
                loss = criterion(outputs, labels)
                loss.backward()


                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            if step % 10 == 0:
                print(f"step {step}: loss={total_loss/len(overfit_loader):.4f} acc={correct/total:.4f}")


    else:

        # Proper training loop
        print("Beginning training...")
        training_data = {
            "training accuracy" : [],
            "training loss" : [],
            "validation accuracy" : [],
            "validation f1" : [],
            "validation loss" : [],
            "time elapsed" : [],
            "peak memory used": [],
            "curr memory used": []
        }
        best_f1 = 0.0
        start_epoch = 0

        # Resume training from last model
        if config["resume"] and os.path.exists("checkpoint.pt"):

            checkpoint = torch.load("checkpoint.pt", map_location=device, weights_only=False)

            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            start_epoch = checkpoint["epoch"] + 1
            best_f1 = checkpoint["best_f1"]
            training_data = checkpoint["training_data"]
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting new training run.")

        end_epoch = start_epoch + config["epochs per run"]

        # Training loop
        f1_no_improvement = 0
        f1_history = deque(maxlen=5)  # rolling window for smoothing
        smoothed_best = 0.0
        PATIENCE = config["patience"]  # break after 30 epochs of no improvement
        l2_lambda = 0.01 
        for epoch in range(start_epoch, end_epoch):
            print(f"\nEpoch {epoch + 1}/{end_epoch}")

            model.train()

            running_loss = 0.0
            train_correct = 0
            train_total = 0

            for windows, labels in train_loader:
                # Move tensors onto CPU/GPU
                windows = windows.to(device)
                labels = labels.to(device)
                # Clear previous gradients
                optimizer.zero_grad()
                outputs = model(windows)
                # loss 
                loss = criterion(outputs, labels) + l2_lambda * l2_reg_loss(model) # small amount of affect l2_loss
                # gradients
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # get Training accuracy
                predictions = torch.argmax(outputs, dim=1).detach()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

            average_train_loss = running_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            print(f"Training Accuracy = {train_accuracy:.4f}")
            training_data["training accuracy"].append(train_accuracy)
            print(f"Training Loss = {average_train_loss:.4f}")
            training_data["training loss"].append(average_train_loss)

            model.eval()

            validation_loss = 0.0

            all_predictions = []
            all_labels = []

            with torch.no_grad():
                start_time = time.perf_counter()
                tracemalloc.start()


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
                
                end_time = time.perf_counter()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

            average_validation_loss = validation_loss / len(validation_loader)

            # Calculate metrics
            accuracy = skm.accuracy_score(all_labels, all_predictions)
            # precision = skm.precision_score(all_labels, all_predictions, average="macro")
            # recall = skm.recall_score(all_labels, all_predictions, average="macro")
            f1 = skm.f1_score(all_labels, all_predictions, average="macro")
            time_elapsed = end_time - start_time
            training_data["validation accuracy"].append(accuracy)
            training_data["validation f1"].append(f1)
            training_data["validation loss"].append(average_validation_loss)
            training_data["time elapsed"].append(time_elapsed)
            training_data["peak memory used"].append(peak / 10**6)
            training_data["curr memory used"].append(current / 10**6)

            print(Counter(all_predictions).most_common(10))
            print(f"Validation Loss = {average_validation_loss:.4f}")
            print(f"Accuracy = {accuracy:.4f}")
            print(f"F1 Score = {f1:.4f}")
            print(f"Time elapsed = {time_elapsed}")
            print(f"Current memory: {current / 10**6:.2f} MB")
            print(f"Peak memory: {peak / 10**6:.2f} MB")
            print(skm.classification_report(all_labels, all_predictions, zero_division=0))


            f1_history.append(f1)
            smoothed_f1 = sum(f1_history) / len(f1_history)

            # Change best f1 if needed
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), "best_model.pt")
                print("New best model saved.")


            if smoothed_f1 > smoothed_best:
                smoothed_best = smoothed_f1
                f1_no_improvement = 0
            else:
                f1_no_improvement += 1
                if f1_no_improvement >= PATIENCE: # patience epochs of no improvement, stop training
                    print(f"No improvement for {PATIENCE} epochs. Stopping training.")
                    break

            # save last model
            torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_f1": best_f1,
            "training_data": training_data,
            "config": config
        }, "checkpoint.pt.tmp") # write to a tmp file then replace
        os.replace("checkpoint.pt.tmp", "checkpoint.pt")


    # Reload best model
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    
    """
    # Final testing
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
