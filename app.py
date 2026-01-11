from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import pickle
from typing import List
import uvicorn

from model import Model
from config import (
    NUM_LAYERS, SIZE_LAYER, DROPOUT_RATE,
    MODEL_PATH, SCALER_PATH
)

app = FastAPI(title="Stock Price Prediction API")

# Global variables for model and scaler
model = None
scaler = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load pre-trained model and scaler."""
    global model, scaler
    try:
        # Load model
        model = Model(NUM_LAYERS, 1, SIZE_LAYER, 1, dropout_rate=DROPOUT_RATE).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        # Load scaler
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        print("Model and scaler loaded successfully")
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Make sure to run main.py first to train and save the model")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# Request/Response models
class PredictionRequest(BaseModel):
    data: List[float]
    
class PredictionResponse(BaseModel):
    prediction: float
    scaled_prediction: float

@app.get("/")
async def root():
    return {"message": "Stock Price Prediction API", "endpoint": "/predict"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a stock price prediction."""
    global model, scaler
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please run training first.")
    
    try:
        # Prepare input data
        data = np.array(request.data, dtype=np.float32).reshape(1, -1, 1)
        input_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        
        # Make prediction
        with torch.no_grad():
            output, _ = model(input_tensor)
            scaled_pred = output.cpu().numpy()[0][0]
        
        # Inverse transform to get original scale
        original_pred = scaler.inverse_transform(np.array([[scaled_pred]]))[0][0]
        
        return PredictionResponse(
            prediction=float(original_pred),
            scaled_prediction=float(scaled_pred)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Check API health and model status."""
    if model is not None and scaler is not None:
        return {"status": "healthy", "model_loaded": True, "device": str(device)}
    return {"status": "unhealthy", "model_loaded": False, "device": str(device)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

