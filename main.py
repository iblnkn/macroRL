#!/usr/bin/env python3
"""
main.py - LSTM-based Inflation Forecasting with TensorBoard Logging
-------------------------------------------------------------------
- Fetch CPI data from FRED (year-over-year inflation).
- Train an LSTM to predict next month's inflation.
- Log all training/test metrics (loss, RMSE, MAE) and final prediction plots to TensorBoard.
"""

import math
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt  # Used to create figures for TensorBoard logging

# ------------------------------ Configurable Hyperparameters ------------------------------
FRED_SERIES = "CPIAUCSL"      # CPI for All Urban Consumers (U.S. city average)
START_DATE = "1980-01-01"
END_DATE = "2022-12-01"
LOOKBACK = 12                 # number of past months for input
TRAIN_FRAC = 0.8              # fraction of data used for training
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3                     # learning rate
DEVICE = "cpu"                # use "cuda" if you have a GPU
# -------------------------------------------------------------------------------------------

def fetch_inflation_data(start=START_DATE, end=END_DATE):
    """
    Fetch monthly CPI from FRED, compute YoY inflation in %, and return as a pandas Series.
    If you have FRED access issues or need an API key, see the earlier notes on manual CSV.
    """
    df_cpi = web.DataReader(FRED_SERIES, 'fred', start, end)
    df_cpi.dropna(inplace=True)
    # Compute year-over-year inflation, i.e., ((CPI[t]/CPI[t-12]) - 1)*100
    df_cpi["inflation"] = df_cpi[FRED_SERIES].pct_change(periods=12) * 100
    df_cpi.dropna(inplace=True)
    return df_cpi["inflation"]

class InflationDataset(Dataset):
    """
    Creates (X, y) sequences from a 1D inflation array:
      - X: the last 'lookback' observations
      - y: the next observation
    """
    def __init__(self, data_array, lookback=LOOKBACK):
        super().__init__()
        self.data = data_array
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        X_seq = self.data[idx : idx + self.lookback]
        y_val = self.data[idx + self.lookback]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

class LSTMForecastModel(nn.Module):
    """
    LSTM regressor for time-series forecasting.
    We interpret (batch_size, lookback) => LSTM with a single feature dimension.
    """
    def __init__(self, hidden_size=32, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, lookback)
        x = x.unsqueeze(-1)  # => (batch, lookback, 1)
        # Initialize hidden/cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))  # => (batch, lookback, hidden_size)
        out = out[:, -1, :]             # last timestep => (batch, hidden_size)
        out = self.fc(out)              # => (batch, 1)
        return out.squeeze(-1)          # => (batch,)

def rmse_func(y_pred, y_true):
    return math.sqrt(np.mean((y_pred - y_true) ** 2))

def mae_func(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def evaluate(model, loader, device="cpu"):
    """
    Returns predictions and ground-truth for the entire dataset in loader.
    We'll compute metrics externally.
    """
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            preds.append(out.cpu().numpy())
            trues.append(y_batch.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return preds, trues

def train_one_epoch(model, loader, optimizer, criterion, device="cpu"):
    model.train()
    running_loss = 0.0
    n_samples = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()

        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        n_samples    += X_batch.size(0)

    return running_loss / n_samples

def main():
    # ------------------------------ 1. Set up TensorBoard writer ---------------------------
    # We append a timestamp so each run has a unique log directory under 'runs/'
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/inflation-{current_time}"
    writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------ 2. Fetch & prep data -----------------------------------
    inflation_ts = fetch_inflation_data()
    data_vals = inflation_ts.values
    n_train = int(len(data_vals) * TRAIN_FRAC)

    train_data = data_vals[:n_train]
    test_data  = data_vals[n_train:]

    train_dataset = InflationDataset(train_data, lookback=LOOKBACK)
    test_dataset  = InflationDataset(test_data,  lookback=LOOKBACK)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # For final plotting, keep track of the date indices
    train_dates = inflation_ts.index[LOOKBACK : n_train]
    test_dates  = inflation_ts.index[n_train + LOOKBACK : ]

    # ------------------------------ 3. Define model & training objects ----------------------
    model = LSTMForecastModel(hidden_size=32, num_layers=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ------------------------------ 4. Training Loop with TensorBoard -----------------------
    for epoch in range(1, EPOCHS+1):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Evaluate on training set
        train_preds, train_trues = evaluate(model, train_loader, DEVICE)
        train_mse = np.mean((train_preds - train_trues)**2)
        train_rmse = rmse_func(train_preds, train_trues)
        train_mae  = mae_func(train_preds, train_trues)

        # Evaluate on test set
        test_preds, test_trues = evaluate(model, test_loader, DEVICE)
        test_mse = np.mean((test_preds - test_trues)**2)
        test_rmse = rmse_func(test_preds, test_trues)
        test_mae  = mae_func(test_preds, test_trues)

        # Log metrics to TensorBoard
        # You can group them with "scalars" in the UI:
        writer.add_scalar("MSE/Train", train_mse, epoch)
        writer.add_scalar("MSE/Test",  test_mse,  epoch)
        writer.add_scalar("RMSE/Train",train_rmse, epoch)
        writer.add_scalar("RMSE/Test", test_rmse,  epoch)
        writer.add_scalar("MAE/Train", train_mae,  epoch)
        writer.add_scalar("MAE/Test",  test_mae,   epoch)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        # Optionally print to console if desired:
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f} | "
              f"Test MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    # ------------------------------ 5. Final Predictions & Plotting to TensorBoard --------------
    # We'll do in-sample (train dataset) & out-of-sample (test dataset) predictions in date order.

    def get_predictions_in_order(model, dataset):
        """
        Get predictions in the original sequential order for the entire dataset
        (use batch_size=1 with no shuffle).
        """
        model.eval()
        preds_list = []
        trues_list = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for X_seq, y_val in loader:
                X_seq = X_seq.to(DEVICE)
                out = model(X_seq)
                preds_list.append(out.item())
                trues_list.append(y_val.item())
        return np.array(preds_list), np.array(trues_list)

    train_preds_seq, train_true_seq = get_predictions_in_order(model, train_dataset)
    test_preds_seq,  test_true_seq  = get_predictions_in_order(model, test_dataset)

    # Let's log final in-sample (train) predictions vs actual as a figure
    fig_train, ax_train = plt.subplots()
    ax_train.plot(train_dates, train_true_seq, label="Train Actual", linestyle="--")
    ax_train.plot(train_dates, train_preds_seq, label="Train Predicted")
    ax_train.set_title("In-Sample (Training) Predictions vs Actual")
    ax_train.set_xlabel("Date")
    ax_train.set_ylabel("Inflation (%)")
    ax_train.legend()
    writer.add_figure("In-Sample_Predictions", fig_train, global_step=EPOCHS)
    plt.close(fig_train)

    # Log final out-of-sample (test) predictions
    fig_test, ax_test = plt.subplots()
    ax_test.plot(test_dates, test_true_seq, label="Test Actual", linestyle="--")
    ax_test.plot(test_dates, test_preds_seq, label="Test Predicted")
    ax_test.set_title("Out-of-Sample (Test) Predictions vs Actual")
    ax_test.set_xlabel("Date")
    ax_test.set_ylabel("Inflation (%)")
    ax_test.legend()
    writer.add_figure("Out-of-Sample_Predictions", fig_test, global_step=EPOCHS)
    plt.close(fig_test)

    # You can log final metrics once more if you like:
    final_train_rmse = rmse_func(train_preds_seq, train_true_seq)
    final_test_rmse  = rmse_func(test_preds_seq,  test_true_seq)
    writer.add_scalar("Final_Metrics/Train_RMSE", final_train_rmse)
    writer.add_scalar("Final_Metrics/Test_RMSE",  final_test_rmse)

    print(f"\nFinal In-Sample (Train) RMSE: {final_train_rmse:.4f}")
    print(f"Final Out-of-Sample (Test) RMSE: {final_test_rmse:.4f}")

    # Flush & close the TensorBoard writer

if __name__ == "__main__":
    main()
