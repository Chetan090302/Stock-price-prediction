import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle
import random

from train import train_forecast
from utils import calculate_accuracy
from config import (
    DATA_PATH, TEST_SIZE, SIMULATION_SIZE,
    MODEL_PATH, SCALER_PATH
)

# Set random seeds for reproducibility
sns.set()
torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Data shape: {df.shape}")
print(df.head())

# Normalize closing prices
minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32'))  # Close index
df_log = minmax.transform(df.iloc[:, 4:5].astype('float32'))    # Close index
df_log = pd.DataFrame(df_log)

# Split data
df_train = df_log.iloc[:-TEST_SIZE]
df_test = df_log.iloc[-TEST_SIZE:]
print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

# Run multiple simulations
print(f"\nRunning {SIMULATION_SIZE} simulations...")
results = []
for i in range(SIMULATION_SIZE):
    print(f'\nSimulation {i + 1}/{SIMULATION_SIZE}')
    results.append(train_forecast(df_train, df_test, df, minmax, device))

# Save scaler for later use
print(f"\nSaving scaler to {SCALER_PATH}...")
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(minmax, f)

# Calculate and display accuracies
accuracies = [
    calculate_accuracy(df['Close'].iloc[-TEST_SIZE:].values, r)
    for r in results
]

print(f"\nAccuracies: {accuracies}")
print(f"Average accuracy: {np.mean(accuracies):.4f}%")
print(f"Std Dev: {np.std(accuracies):.4f}%")

# Plot results
plt.figure(figsize=(15, 5))
for no, r in enumerate(results):
    plt.plot(r, label=f'forecast {no + 1}')
plt.plot(df['Close'].iloc[-TEST_SIZE:].values, label='true trend', c='black', linewidth=2)
plt.legend()
plt.title(f'Stock Price Prediction - Average Accuracy: {np.mean(accuracies):.4f}%')
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid(True, alpha=0.3)
plt.show()

print("\nTraining complete! Model and scaler saved.")
print(f"Model: {MODEL_PATH}")
print(f"Scaler: {SCALER_PATH}")