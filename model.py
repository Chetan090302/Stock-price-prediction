import torch
import torch.nn as nn

class Model(nn.Module):
    """
    LSTM-based neural network for stock price prediction.
    
    Args:
        num_layers (int): Number of LSTM layers
        size (int): Input feature size
        size_layer (int): Hidden layer size
        output_size (int): Output size
        dropout_rate (float): Dropout probability
    """
    def __init__(
        self,
        num_layers,
        size,
        size_layer,
        output_size,
        dropout_rate=0.1,
    ):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.size_layer = size_layer
        self.lstm = nn.LSTM(
            size,
            size_layer,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.fc = nn.Linear(size_layer, output_size)
        
    def forward(self, x, hidden_state=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden_state: Previous hidden state (optional)
            
        Returns:
            logits: Model predictions
            hidden_state: New hidden state for next sequence
        """
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(x)
        else:
            lstm_out, hidden_state = self.lstm(x, hidden_state)
        logits = self.fc(lstm_out[:, -1, :])
        return logits, hidden_state
