# --- Standard library ---
import os
import time
import random
import statistics
import tracemalloc
from collections import Counter, deque

# --- Third‑party ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn.metrics as skm
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# --- Local modules ---
from config import parse_args, create_config
from src.datasets import PMUData, PMUDataset
from src.models import gpt4ts
from src.visualise import plot_confusion_matrix


# from GPT4TS
def l2_reg_loss(model):
    for name, param in model.named_parameters():
        if name == 'out_layer.weight':  # match YOUR layer's actual name
            return torch.sum(torch.square(param))

    
def run(config):
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
    if config["experiment"]: 
        print("Loading training data...")
        train_data = PMUData(config['train data'], config['train pattern'], config)
        train_dataset = PMUDataset(train_data)
        labels = train_dataset.labels_df['Label'].values
        per_class_cap = 100

        selected_indices = []
        for cls in np.unique(labels):
            cls_indices = np.where(labels == cls)[0]
            selected_indices.extend(cls_indices[:per_class_cap].tolist())

        overfit_dataset = Subset(train_dataset, selected_indices)
        overfit_loader = DataLoader(overfit_dataset, batch_size=32, shuffle=True)

        # small validation slice for the experiment
        print("Loading validation data...")
        validation_data = PMUData(config['validation data'], config['validation pattern'], config)
        validation_dataset = PMUDataset(validation_data)
        val_labels = validation_dataset.labels_df['Label'].values
        val_per_class_cap = 20  # keep small

        val_selected_indices = []
        for cls in np.unique(val_labels):
            cls_indices = np.where(val_labels == cls)[0]
            val_selected_indices.extend(cls_indices[:val_per_class_cap].tolist())

        overfit_val_dataset = Subset(validation_dataset, val_selected_indices)
        overfit_val_loader = DataLoader(overfit_val_dataset, batch_size=32, shuffle=False)
    else:
        if not config["test only"]:
            print("Loading training data...")
            train_data = PMUData(config['train data'], config['train pattern'], config)
            print("Loading validation data...")
            validation_data = PMUData(config['validation data'], config['validation pattern'], config)
            train_dataset = PMUDataset(train_data)
            print(f"Train dataset length: {len(train_dataset.labels_df)}")
            validation_dataset = PMUDataset(validation_data)

            #Create Dataloaders (create mini batches)
            train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=config['batch size'],      # Group data into chunks of 32
            shuffle=True,       # Mix up data order every epoch
            num_workers=2,      # Use 2 CPU subprocesses to load data parallelly
            pin_memory=True     # Speed up data copy to GPU memory
            )

            # Make validation set smaller samples to increase epoch speed
            val_samples_per_class = 100 # can tune
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

        print("Loading testing data")
        test_data = PMUData(config['test data'], config['test pattern'], config)

        # create PMUDataset object wrappers
        
        test_dataset = PMUDataset(test_data)

        test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=config['batch size'],      
        shuffle=True,       
        num_workers=2,      
        pin_memory=True
        )

    if not config["test only"]:
        # Create model
        print(f"Initialising {config['model']} Model...")
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
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": config["backbone learning rate"], "weight_decay": 0.0,},
            {"params": head_params, "lr": config["head learning rate"], "weight_decay": config["head weight decay"],},
        ])
        
        for i, g in enumerate(optimizer.param_groups):
            print(f"group {i}: lr={g['lr']}, num_params={sum(p.numel() for p in g['params'])}")

        #  Learning rate scheduler
        if config["scheduler"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        elif config["scheduler"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs per run"], eta_min=config["backbone learning rate"] * 0.01)
        else:
            scheduler = None
        print(f"Using scheduler: {config['scheduler']}")
        # Loss model
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # for classification, 0.1 prevents over confidence

    l2_lambda = config["l2 lambda"]
    # experiment loop
    if config["experiment"]:
        print("Beginning experimental training")
        for step in range(config["epochs per run"]):
            correct, total, total_loss = 0, 0, 0.0
            model.train()
            for windows, labels in overfit_loader:
                windows, labels = windows.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(windows)
                loss = criterion(outputs, labels) + l2_lambda * l2_reg_loss(model)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)


            if step % 10 == 0 or config["epochs per run"] <= 100:
                model.eval()
                val_correct, val_total, val_loss = 0, 0, 0.0
                with torch.no_grad():
                    for windows, labels in overfit_val_loader:
                        windows, labels = windows.to(device), labels.to(device)
                        outputs = model(windows)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        preds = torch.argmax(outputs, dim=1)
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)
                model.train()

                print(f"step {step}: train_loss={total_loss/len(overfit_loader):.4f} "
                    f"train_acc={correct/total:.4f} "
                    f"val_loss={val_loss/len(overfit_val_loader):.4f} "
                    f"val_acc={val_correct/val_total:.4f}")
        return

    elif not config["test only"]:

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
            if config["scheduler"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])

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
            avg_time = time_elapsed / len(validation_loader.dataset)
            cm = skm.confusion_matrix(all_labels, all_predictions)
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
            print(cm)

            if config["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(f1)
            elif config["scheduler"] == "CosineAnnealingLR":
                scheduler.step()
            
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
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_f1": best_f1,
                "training_data": training_data,
                "config": config,
            }

            if scheduler is not None:
                checkpoint["scheduler"] = scheduler.state_dict()

            torch.save(checkpoint, "checkpoint.pt.tmp")
            os.replace("checkpoint.pt.tmp", "checkpoint.pt")


    # AFTER TRAINING, TESTING - Reload best model
    model = gpt4ts(config, test_data)
    model.to(device)
    model.load_state_dict(torch.load("models/bert/FREQUAUBUCIAIBICglobal/best_model.pt", map_location=device))
    model.eval()

    all_predictions = []
    all_labels = []
    classification_times = []
    peak_memories = []
    current_memories = []

    # Warm-up: run a few throwaway inferences to avoid measuring one-time setup cost
    with torch.no_grad():
        warm_up_batch, _ = next(iter(test_loader))
        warm_up_window = warm_up_batch[0:1].to(device)
        for _ in range(5):
            _ = model(warm_up_window)
            if device.type == "cuda":
                torch.cuda.synchronize()

    with torch.no_grad():
        for windows, labels in test_loader:
            labels = labels.to(device)
            batch_predictions = []

            for i in range(windows.size(0)):
                window = windows[i:i+1].to(device)

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                start = time.perf_counter()
                outputs = model(window)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()

                classification_times.append(end - start)

                if device.type == "cuda":
                    peak_memories.append(torch.cuda.max_memory_allocated() / 1024**2)
                    current_memories.append(torch.cuda.memory_allocated() / 1024**2)

                prediction = torch.argmax(outputs, dim=1)
                batch_predictions.append(prediction.item())

            all_predictions.extend(batch_predictions)
            all_labels.extend(labels.cpu().numpy())
    
    # Compute final metrics

    accuracy = skm.accuracy_score(all_labels, all_predictions)
    precision = skm.precision_score(all_labels, all_predictions, average="macro")
    recall = skm.recall_score(all_labels, all_predictions, average="macro")
    f1 = skm.f1_score(all_labels, all_predictions, average="macro")

    print("Final Results")
    print("---------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Mean single-window inference time: {statistics.mean(classification_times)*1000:.3f} ms")
    if device.type == "cuda":
        print(f"Mean peak memory per window: {statistics.mean(peak_memories):.2f} MB")

    plot_confusion_matrix(all_labels, all_predictions, save_path="confusion_matrix.png")
        

if __name__ == "__main__":
    args = parse_args()
    config = create_config(args)
    print(f"Columns config: {config['columns']}")
    run(config)
