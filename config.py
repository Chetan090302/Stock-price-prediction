# Configuration and Hyperparameters
# Model Architecture
NUM_LAYERS = 1
SIZE_LAYER = 128
DROPOUT_RATE = 0.8

# Training Parameters
EPOCH = 300
LEARNING_RATE = 0.01
TIMESTAMP = 5

# Data Parameters
TEST_SIZE = 30
SIMULATION_SIZE = 10

# File Paths
MODEL_PATH = 'trained_model.pt'
SCALER_PATH = 'minmax_scaler.pkl'
DATA_PATH = '/Users/jettychetan/Desktop/crud/Stock-price-prediction/GOOG-year.csv'

# Signal Smoothing
ANCHOR_WEIGHT = 0.3
