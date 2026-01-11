import numpy as np

def calculate_accuracy(real, predict):
    """
    Calculate accuracy using RMSE-based metric.
    
    Args:
        real: Actual values
        predict: Predicted values
        
    Returns:
        float: Accuracy percentage (0-100)
    """
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100

def anchor(signal, weight):
    """
    Apply exponential smoothing to signal.
    
    Args:
        signal: Input signal/array
        weight: Smoothing weight (0-1)
        
    Returns:
        list: Smoothed signal
    """
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer
