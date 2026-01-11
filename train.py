import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

from model import Model
from utils import calculate_accuracy, anchor
from config import (
    NUM_LAYERS, SIZE_LAYER, EPOCH, DROPOUT_RATE, 
    LEARNING_RATE, TIMESTAMP, TEST_SIZE, ANCHOR_WEIGHT,
    MODEL_PATH, SCALER_PATH
)

def train_forecast(df_train, df_test, df, minmax, device):
    """
    Train the model and generate forecasts.
    
    Args:
        df_train: Training data
        df_test: Test data
        df: Original dataframe (for dates)
        minmax: MinMaxScaler instance
        device: torch device (cuda or cpu)
        
    Returns:
        list: Forecasted prices for test period
    """
    # Initialize model
    model = Model(
        NUM_LAYERS,
        df_train.shape[1],
        SIZE_LAYER,
        df_train.shape[1],
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
    
    # Training loop
    pbar = tqdm(range(EPOCH), desc='train loop')
    for i in pbar:
        total_loss, total_acc = [], []
        hidden_state = None
        
        for k in range(0, df_train.shape[0] - 1, TIMESTAMP):
            index = min(k + TIMESTAMP, df_train.shape[0] - 1)
            batch_x = torch.tensor(
                df_train.iloc[k:index, :].values,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)
            batch_y = torch.tensor(
                df_train.iloc[k + 1:index + 1, :].values,
                dtype=torch.float32,
                device=device
            )
            
            logits, hidden_state = model(batch_x, hidden_state)
            hidden_state = tuple(h.detach() for h in hidden_state) if hidden_state else None
            
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss.append(loss.item())
            total_acc.append(calculate_accuracy(
                batch_y[:, 0].cpu().detach().numpy(),
                logits[:, 0].cpu().detach().numpy()
            ))
        
        pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))
    
    # Save model after training
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Generate predictions
    future_day = TEST_SIZE
    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]
    upper_b = (df_train.shape[0] // TIMESTAMP) * TIMESTAMP
    hidden_state = None
    
    with torch.no_grad():
        # Predictions on training data
        for k in range(0, (df_train.shape[0] // TIMESTAMP) * TIMESTAMP, TIMESTAMP):
            batch_x = torch.tensor(
                df_train.iloc[k:k + TIMESTAMP, :].values,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)
            out_logits, hidden_state = model(batch_x, hidden_state)
            output_predict[k + 1:k + TIMESTAMP + 1] = out_logits.cpu().numpy()
        
        # Handle remaining training data
        if upper_b != df_train.shape[0]:
            batch_x = torch.tensor(
                df_train.iloc[upper_b:, :].values,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)
            out_logits, hidden_state = model(batch_x, hidden_state)
            output_predict[upper_b + 1:df_train.shape[0] + 1] = out_logits.cpu().numpy()
            future_day -= 1
            date_ori.append(date_ori[-1] + timedelta(days=1))
        
        # Future predictions
        for i in range(future_day):
            o = output_predict[-future_day - TIMESTAMP + i:-future_day + i]
            batch_x = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            out_logits, hidden_state = model(batch_x, hidden_state)
            output_predict[-future_day + i] = out_logits.cpu().numpy()[-1]
            date_ori.append(date_ori[-1] + timedelta(days=1))
    
    # Inverse transform predictions
    output_predict = minmax.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], ANCHOR_WEIGHT)
    
    return deep_future[-TEST_SIZE:]
