import torch
import torch.nn as nn

class ImprovedLSTMModel(nn.Module):
    """
    An improved LSTM model for stock price prediction with bidirectionality, dropout,
    and skip connections for better gradient flow.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.2, bidirectional=True):
        """
        Initialize the LSTM model with configurable parameters
        
        Args:
            input_dim (int): Input dimension (number of features)
            hidden_dim (int): Hidden dimension of the LSTM layers
            num_layers (int): Number of LSTM layers
            output_dim (int): Output dimension (usually 1 for price prediction)
            dropout (float): Dropout rate for regularization
            bidirectional (bool): Whether the LSTM is bidirectional
        """
        super(ImprovedLSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # Define the LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension is doubled for bidirectional LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism for focusing on important time steps
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature projection layers
        self.proj1 = nn.Linear(lstm_output_dim, 128)
        self.proj2 = nn.Linear(input_dim, 32)  # Skip connection from input
        
        # Final prediction layers
        self.fc1 = nn.Linear(128 + 32, 64)  # Combining the skip connection
        self.fc2 = nn.Linear(64, output_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        
        # Batch normalization for improved training stability
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        """
        Forward pass of the LSTM model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Predicted output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Save input for the skip connection
        input_skip = x[:, -1, :]  # Use the last time step
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch, seq_len, hidden_dim*2)
        
        # Attention mechanism to focus on important time steps
        attention_weights = self.attention(lstm_out)  # shape: (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # shape: (batch, hidden_dim*2)
        
        # Process the LSTM output
        lstm_features = self.proj1(context_vector)
        lstm_features = self.bn1(lstm_features)
        lstm_features = self.relu(lstm_features)
        lstm_features = self.dropout_layer(lstm_features)
        
        # Process the skip connection
        skip_features = self.proj2(input_skip)
        skip_features = self.bn2(skip_features)
        skip_features = self.relu(skip_features)
        
        # Combine features from LSTM and skip connection
        combined = torch.cat((lstm_features, skip_features), dim=1)
        
        # Final prediction
        out = self.fc1(combined)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        return out
    
    def __str__(self):
        """String representation of the model with its key parameters"""
        return (f"ImprovedLSTMModel(input_dim={self.lstm.input_size}, "
                f"hidden_dim={self.hidden_dim}, "
                f"num_layers={self.num_layers}, "
                f"output_dim={self.fc2.out_features}, "
                f"dropout={self.dropout}, "
                f"bidirectional={self.bidirectional})")

# Function to create LSTM models for ensemble
def create_lstm_ensemble(input_dim, hidden_dims, num_layers_list, output_dim):
    """Create an ensemble of LSTM models with different architectures"""
    models = []
    
    for hidden_dim in hidden_dims:
        for num_layers in num_layers_list:
            # Standard bidirectional LSTM with attention
            models.append(ImprovedLSTMModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                bidirectional=True
            ))
            
            # Unidirectional LSTM (for diversity)
            models.append(ImprovedLSTMModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                bidirectional=False
            ))
    
    return models 