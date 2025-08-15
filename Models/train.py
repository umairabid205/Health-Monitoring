# test_dataloader_and_model.py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
from torch.utils.data import DataLoader, TensorDataset

# Import the model (assuming Models/models.py has HybridModel class)
# at the top of your file
import sys
import os

# Go two levels up from this file to reach project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from Models.models import HybridModel  # adjust path if needed

# ==== Choose dataset for test ====
# We'll use v1 first
# Make sure you run this after the prepare_X_as_3d/prepare_y step
# or re-import X1, y1 from saved npy files
import numpy as np
import pandas as pd
# Example: reload from numpy if needed:
X1 = np.load("Data/processed/X3_features.npy")
y1 = np.load("Data/Processed/y3_labels.npy")

X = torch.tensor(X1, dtype=torch.float32)
y = torch.tensor(y1, dtype=torch.long)


#Split 
X_train, X_val , y_train ,y_val = train_test_split(X, y , test_size = 0.2 , stratify=y , random_state = 42)

# Create DataLoader with TesorDataset
batch_size = 8
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val , y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset , batch_size=batch_size , shuffle=False)


# ==== Model Run ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridModel(
    input_channels=X.shape[2],  # channels
    cnn_channels=32,
    lstm_hidden=64,
    lstm_layers=1,
    num_classes=len(torch.unique(y))
).to(device)


# Early Stopping Helper
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        
    def __call__(self, val_loss):
        score = -val_loss  # we minimize val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# Training Loop
def train_with_early_stopping(model, train_loader, val_loader , device , epochs = 50 , patience=5):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopper = EarlyStopping(patience=patience)
    
    
    train_losses =[]
    val_losses = []
    
    
    for epoch in range(epochs):
        
        # ----------Training----------
        model.train()
        running_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        epoch_train_loss =running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        
        
        # --- Validation ---
        model.eval()
        val_running_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_running_loss += loss.item() * X_batch.size(0)

                preds = torch.argmax(outputs, dim=1)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
        
        # Early stopping check
        early_stopper(epoch_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
        
        
        
            # --- Plot Loss ---
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.show()

    return model




# model = HybridModel(input_channels=1, cnn_channels=32, lstm_hidden=64, lstm_layers=1, num_classes=4).to(device)
train_with_early_stopping(model, train_loader, val_loader, device, epochs=50, patience=7)

# Train for 3 epochs

# for epoch in range(100):
#     train_loss = train_one_epoch_simple(model, train_loader, criterion, optimizer, device)
#     y_true_val, y_pred_val = evaluate(model, val_loader, device)
    
#     acc = accuracy_score(y_true_val, y_pred_val)
#     prec = precision_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
#     rec = recall_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
#     f1 = f1_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
    
#     print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

















