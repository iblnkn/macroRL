#!/usr/bin/env python3
"""
A Minimal LSTM Example for Inflation Forecasting in Economics
-------------------------------------------------------------
What it does:
  1) Fetches monthly CPI data from FRED, computes YOY inflation (%).
  2) Creates sequences of length 'lookback' to predict next-month inflation.
  3) Normalizes the inflation series (mean=0, std=1).
  4) Defines and trains a single-layer LSTM in PyTorch for forecasting.
  5) Plots the predicted vs. actual inflation on both training and test sets.

"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import datetime

# ----------------- Hyperparameters ------------------
LOOKBACK      = 24      # months of historical data used as input
BATCH_SIZE    = 32      # mini-batch size
EPOCHS        = 50      # number of training epochs
LEARNING_RATE = 1e-3    # learning rate for the optimizer
HIDDEN_SIZE   = 64      # LSTM hidden dimension
DROP_PROB     = 0.1     # dropout probability
TRAIN_FRAC    = 0.8     # fraction of data to use for training
START_DATE    = "1980-01-01"
END_DATE      = "2022-12-01"

# ------------------- Data Utilities -------------------
def fetch_cpi_from_fred(fred_api_key=None):
    """
    Fetch CPI data (monthly) from FRED. 
    If you have an API key, pass it in; otherwise, it usually works without one.
    Returns a Series of CPI with a monthly DateTime index.
    """
    series_name = "CPIAUCSL"  # CPI All Urban Consumers
    if fred_api_key:
        df = web.DataReader(series_name, "fred", START_DATE, END_DATE, api_key=fred_api_key)
    else:
        df = web.DataReader(series_name, "fred", START_DATE, END_DATE)
    # Ensure monthly frequency (sometimes daily if a key is used)
    df = df.resample("MS").mean().ffill()
    return df[series_name]

def compute_yoy_inflation(cpi_series):
    """
    Given a monthly CPI series, compute the year-over-year inflation rate (percent).
    inflation_t = (CPI_t / CPI_{t-12} - 1) * 100.
    Returns a pandas Series of inflation rates.
    """
    inflation = (cpi_series.pct_change(12) * 100).dropna()
    return inflation

def make_sequences(inflation_series, lookback=12):
    """
    Build a 2D numpy array of shape (N, lookback) for the LSTM input (X)
    and a 1D array (N,) for the next-month inflation (y).
    
    For time index t, X_t contains inflation at t-lookback,...,t-1.
    y_t is inflation at time t.
    """
    values = inflation_series.values  # 1D array of shape (length,)
    sequences = []
    targets = []
    for i in range(len(values) - lookback):
        X_seq = values[i : i + lookback]
        y_val = values[i + lookback]
        sequences.append(X_seq)
        targets.append(y_val)
    return np.array(sequences), np.array(targets)

# ----------------- PyTorch Dataset & Model -----------------
class InflationDataset(Dataset):
    """
    Simple Dataset: each item is (X, y)
      X = shape (lookback,)  (we'll expand dims later for LSTM)
      y = scalar inflation
    """
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # LSTM expects shape (batch, sequence_length, input_dim).
        # Here input_dim=1 since we only have inflation as a feature.
        X_seq = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(-1)
        y_val = torch.tensor(self.y[idx], dtype=torch.float32)
        return X_seq, y_val

class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size=32, dropout=0.0):
        """
        A single-layer LSTM that outputs 1 value (inflation).
        """
        super().__init__()
        self.hidden_size = hidden_size
        # input_size=1 because we have only 1 feature (inflation)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: shape (batch_size, lookback, 1)
        """
        # h_0 and c_0 default to zeros if not provided
        out, _ = self.lstm(x)      # out => (batch_size, lookback, hidden_size)
        out = out[:, -1, :]        # get the last time-step
        out = self.dropout(out)
        out = self.fc(out)         # => (batch_size, 1)
        return out.squeeze(-1)     # => (batch_size,)

# ----------------- Training & Plotting -----------------
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(train_loader.dataset)

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            all_preds.append(preds.numpy())
            all_targets.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)

def main():
    # 1) Fetch CPI data and compute inflation
    cpi = fetch_cpi_from_fred(fred_api_key=None)
    inflation = compute_yoy_inflation(cpi)
    
    # 2) Split data into train/test
    n_total = len(inflation)
    n_train = int(n_total * TRAIN_FRAC)
    train_inflation = inflation.iloc[:n_train]
    test_inflation  = inflation.iloc[n_train:]
    
    # 3) Normalize each split (basic standardization)
    mean_train = train_inflation.mean()
    std_train  = train_inflation.std()
    train_scaled = (train_inflation - mean_train) / std_train
    test_scaled  = (test_inflation - mean_train) / std_train
    
    # 4) Build sequences for model input
    X_train, y_train = make_sequences(train_scaled, lookback=LOOKBACK)
    X_test, y_test   = make_sequences(test_scaled,  lookback=LOOKBACK)
    
    # Note: we lose 'LOOKBACK' points at the start of each set.
    train_dates = train_inflation.index[LOOKBACK:]
    test_dates  = test_inflation.index[LOOKBACK:]
    
    # 5) Dataloaders
    train_dataset = InflationDataset(X_train, y_train)
    test_dataset  = InflationDataset(X_test,  y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
    
    # 6) Define model, optimizer, loss
    model = SimpleLSTM(hidden_size=HIDDEN_SIZE, dropout=DROP_PROB)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 7) Training loop
    for epoch in range(1, EPOCHS+1):
        train_loss = train_model(model, train_loader, optimizer, criterion)
        # Evaluate on test set
        test_preds, test_targets = evaluate_model(model, test_loader)
        test_loss = np.mean((test_preds - test_targets)**2)
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
    
    # 8) Predictions & Plot
    # Predict on entire train and test sets
    train_preds, train_targs = evaluate_model(model, train_loader)
    test_preds,  test_targs  = evaluate_model(model, test_loader)
    
    # Invert the normalization
    train_preds_real = train_preds * std_train + mean_train
    train_targs_real = train_targs * std_train + mean_train
    test_preds_real  = test_preds  * std_train + mean_train
    test_targs_real  = test_targs  * std_train + mean_train
    
    # Convert to DataFrame for easy plotting
    df_train_plot = pd.DataFrame({
        "Date": train_dates[len(train_preds_real)*-1:],  # align last part
        "Predicted": train_preds_real,
        "Actual": train_targs_real
    }).set_index("Date")
    
    df_test_plot = pd.DataFrame({
        "Date": test_dates[len(test_preds_real)*-1:],  # align last part
        "Predicted": test_preds_real,
        "Actual": test_targs_real
    }).set_index("Date")
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_train_plot.index, df_train_plot["Actual"], label="Train Actual", color="blue")
    plt.plot(df_train_plot.index, df_train_plot["Predicted"], label="Train Predicted", color="red", alpha=0.7)
    plt.title("LSTM Inflation Forecast - Training Set")
    plt.ylabel("Inflation (YOY %)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df_test_plot.index, df_test_plot["Actual"], label="Test Actual", color="blue")
    plt.plot(df_test_plot.index, df_test_plot["Predicted"], label="Test Predicted", color="red", alpha=0.7)
    plt.title("LSTM Inflation Forecast - Test Set")
    plt.ylabel("Inflation (YOY %)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
