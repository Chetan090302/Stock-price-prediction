# Stock Price Prediction using LSTM

A PyTorch-based LSTM model for predicting stock prices using historical closing price data. The project includes a FastAPI REST API for making predictions.

## Project Overview

This project trains an LSTM (Long Short-Term Memory) neural network to predict Google stock prices. The model learns patterns from historical data and can forecast future price movements.

## Features

- **LSTM-based Model**: Uses a recurrent neural network for time-series prediction
- **Data Preprocessing**: MinMax scaling for normalization
- **Ensemble Predictions**: Multiple simulations for robust predictions
- **REST API**: FastAPI endpoint for real-time predictions
- **Model Persistence**: Saves trained models and scalers for reuse

## Project Structure

```
Stock-price-prediction/
├── main.py                    # Main training script
├── app.py                     # FastAPI server for predictions
├── config.py                  # Configuration and hyperparameters
├── model.py                   # LSTM Model definition
├── train.py                   # Training logic
├── utils.py                   # Utility functions
├── GOOG-year.csv             # Historical Google stock data
├── trained_model.pt          # Saved model weights (generated)
├── minmax_scaler.pkl         # Saved scaler (generated)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- FastAPI >= 0.104.0
- uvicorn >= 0.24.0

## Installation

1. Clone or download the project:
```bash
cd Stock-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

Run the training script to train the LSTM model:

```bash
python main.py
```

This will:
- Load the GOOG-year.csv dataset
- Normalize data using MinMaxScaler
- Train the model for 300 epochs
- Run 10 simulations for ensemble predictions
- Save the trained model to `trained_model.pt`
- Save the scaler to `minmax_scaler.pkl`
- Display accuracy metrics and predictions plot

### 2. Running the Prediction API

Start the FastAPI server:

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### 3. Making Predictions

#### Check API Health
```bash
curl http://localhost:8000/health
```

#### Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [0.5, 0.45, 0.48, 0.52, 0.50]}'
```

Response:
```json
{
  "prediction": 150.25,
  "scaled_prediction": 0.5023
}
```

#### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation

## Model Architecture

The LSTM model consists of:
- **Input Layer**: Takes normalized closing prices
- **LSTM Layer**: 1 layer with 128 hidden units
- **Dense Layer**: Fully connected layer for output prediction
- **Dropout**: 0.8 dropout rate for regularization

## Configuration

Key hyperparameters in `config.py`:
- `num_layers`: 1 (number of LSTM layers)
- `size_layer`: 128 (number of hidden units)
- `timestamp`: 5 (sequence length)
- `epoch`: 300 (training epochs)
- `dropout_rate`: 0.8 (dropout probability)
- `learning_rate`: 0.01 (Adam optimizer learning rate)
- `test_size`: 30 (test set size)
- `simulation_size`: 10 (number of ensemble simulations)

## Accuracy Metrics

The model uses **RMSE-based accuracy** calculation:
```
Accuracy = (1 - sqrt(mean((real - predict)² / real²))) × 100%
```

Higher accuracy indicates better predictions.

## File Descriptions

### main.py
Main entry point for training. Orchestrates data loading, model training, and evaluation.

### app.py
FastAPI application providing REST endpoints for predictions.

### config.py
Configuration constants and hyperparameters.

### model.py
LSTM model class definition using PyTorch.

### train.py
Training loop logic and forecast function.

### utils.py
Utility functions for accuracy calculation, signal smoothing, etc.

## Data Format

### Input Data (GOOG-year.csv)
Expected columns:
- Date
- Open
- High
- Low
- Close (used for prediction)
- Volume

### API Prediction Input
List of normalized (0-1) closing prices:
```json
{
  "data": [0.5, 0.45, 0.48, 0.52, 0.50]
}
```

## Device Support

The model automatically detects and uses:
- GPU (CUDA) if available for faster training
- CPU as fallback

## Troubleshooting

### Model not found error
Make sure to run `python main.py` first to train and save the model.

### Port 8000 already in use
Change the port in app.py:
```python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Out of memory
Reduce `size_layer` or `simulation_size` in config.py.

## Future Enhancements

- [ ] Add multiple stock symbols support
- [ ] Implement more evaluation metrics
- [ ] Add model versioning
- [ ] Create frontend dashboard
- [ ] Add data augmentation techniques
- [ ] Implement model explainability (SHAP)

## License

This project is open source and available under the MIT License.

## References

- PyTorch LSTM Documentation: https://pytorch.org/docs/stable/nn.html#lstm
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Time Series Forecasting: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
