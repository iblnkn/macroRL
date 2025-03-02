# macroRL: LSTM-Based Macroeconomic Forecasting with PyTorch

This repository contains a toy project for forecasting macroeconomic variables (e.g., CPI-based inflation) using an LSTM in PyTorch. It includes TensorBoard logging for visualization, and fetches data from [FRED](https://fred.stlouisfed.org/) via `pandas_datareader`.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Using Miniconda (Recommended on Windows)](#installation-using-miniconda-recommended-on-windows)
3. [Usage](#usage)
4. [Running TensorBoard](#running-tensorboard)
5. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
6. [License](#license)

---

## Prerequisites

- **Miniconda** (or Anaconda) on Windows, macOS, or Linux.
- Basic familiarity with Python, PyTorch, and Git.

If you prefer a plain Python installation without Conda, you must manually manage your environment and ensure all dependencies are installed (PyTorch, pandas, numpy, etc.). Using Conda simplifies environment management and avoids version conflicts.

---

## Installation Using Miniconda (Recommended on Windows)

### 1. Install Miniconda

- Download Miniconda for your platform from:  
  [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- For Windows, choose “Windows 64-bit” (the default on most modern systems).
- Double-click to install, following the prompts. You can keep the default settings.

### 2. Launch the Miniconda Prompt

- On Windows, open your Start Menu and look for "Miniconda Prompt" (or "Anaconda Prompt").
- This special prompt ensures `conda` is recognized.

### 3. Clone or Download this Repository

- If you have Git installed, open the Miniconda Prompt and run:
  ```powershell
  git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
  cd YOUR_REPO
  ```
- Or download the repo as a ZIP from GitHub and extract it, then `cd` into the unzipped folder in the Miniconda Prompt:
  ```powershell
  cd path\to\YOUR_REPO
  ```

### 4. Create a Conda Environment

```powershell
conda create -n macroRL python=3.10
conda activate macroRL
```
- This creates an environment named `macroRL` with Python 3.10.  
- Adjust the Python version if you wish (e.g. `3.9`, `3.11`).

### 5. Install Dependencies

- **CPU-Only PyTorch** (simplest if you don’t need GPU):
  ```powershell
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  ```

- **NVIDIA GPU + CUDA** (for faster training):
  ```powershell
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
  > This automatically installs CUDA 11.8 support with PyTorch.

- **Other libraries**:
  ```powershell
  conda install pandas numpy pandas-datareader
  pip install tensorboard
  ```
  (Optionally add `conda install matplotlib` if you want local plots.)

### 6. Verify Your Environment

- Check Python:
  ```powershell
  python --version
  ```
- Check PyTorch version:
  ```powershell
  python -c "import torch; print(torch.__version__)"
  ```
- Check TensorBoard import:
  ```powershell
  python -c "import tensorboard"
  ```
If these commands run without errors, you’re set.

---

## Usage

1. **Activate the Environment**  
   In a new terminal (Miniconda Prompt):
   ```powershell
   conda activate macroRL
   ```
   Then navigate to the project folder:
   ```powershell
   cd path\to\YOUR_REPO
   ```

2. **Run the Training Script**  
   ```powershell
   python main.py
   ```
   - This script will:
     1. Fetch CPI data from FRED (via `pandas_datareader`).
     2. Compute year-over-year inflation.
     3. Train an LSTM model in PyTorch.
     4. Log metrics (loss, RMSE, etc.) and plots to TensorBoard in a `runs/` directory.

3. **(Optional) Adjust Hyperparameters**  
   - Open `main.py` and change parameters like `EPOCHS`, `LOOKBACK`, or the LSTM `hidden_size`.  
   - Rerun `python main.py` to see how it affects training.

---

## Running TensorBoard

To view logs in real-time or after training completes:

```powershell
conda activate macroRL
cd path\to\YOUR_REPO
tensorboard --logdir=runs --port=6006
```
If `tensorboard` isn’t recognized, try:

```powershell
python -m tensorboard --logdir=runs --port=6006
```

Open [http://localhost:6006](http://localhost:6006) in your browser.  
You’ll see:

- **Scalars**: Training/validation losses, RMSE, MAE across epochs  
- **Figures**: Plots of actual vs. predicted inflation (if your script logs them)  
- **Graphs**: High-level model graph (optional)

---

## Common Issues and Troubleshooting

1. **`conda: command not found` or “Term 'conda' is not recognized”**  
   - On Windows, ensure you’re in the **Miniconda Prompt** (Start Menu → “Miniconda3 Prompt”) or have run `conda init powershell` so that conda is recognized in your shell.
   - Restart your terminal/PowerShell if changes have been made to PATH.

2. **`No module named tensorboard`** or “`tensorboard not recognized`”**  
   - Install TensorBoard again with:
     ```powershell
     conda activate macroRL
     pip install tensorboard
     ```
   - Or run with `python -m tensorboard --logdir=runs`.
   - Confirm you’re in the correct conda environment.

3. **FRED Data Access Problems**  
   - If `pandas_datareader` cannot fetch data due to rate limits, you may need an API key. See [FRED API docs](https://fred.stlouisfed.org/docs/api/api_key.html).
   - Or manually download a CSV from FRED and load it in `main.py` instead.

4. **GPU Not Detected**  
   - Check if your GPU is recognized by PyTorch:
     ```powershell
     python -c "import torch; print(torch.cuda.is_available())"
     ```
   - If `False`, ensure you installed the GPU version (`pytorch-cuda=11.8`) and have compatible NVIDIA drivers.
   - Make sure your script sets `DEVICE = "cuda"` (if `torch.cuda.is_available()` returns True).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

