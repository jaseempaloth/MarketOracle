import torch
import torch.nn as nn

class ImprovedGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2, bidirectional=True):
        super(ImprovedGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.directions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers with residual connection
        self.fc1 = nn.Linear(hidden_dim * self.directions, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_dim).to(x.device)
        
        # Forward pass through GRU
        gru_out, _ = self.gru(x, h0)  # gru_out: [batch_size, seq_len, hidden_dim * directions]
        
        # Apply attention
        attn_weights = self.attention(gru_out)  # [batch_size, seq_len, 1]
        context = torch.bmm(attn_weights.transpose(1, 2), gru_out)  # [batch_size, 1, hidden_dim * directions]
        context = context.squeeze(1)  # [batch_size, hidden_dim * directions]
        
        # Output layers with residual connection
        out = self.fc1(context)
        residual = out  # Save for residual connection
        out = self.dropout(out)
        out = self.relu(out)
        out = out + residual  # Residual connection
        out = self.fc2(out)
        
        return out

class CustomLoss(nn.Module):
    def __init__(self, direction_weight=0.3):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.direction_weight = direction_weight
        
    def forward(self, y_pred, y_true):
        # Standard MSE loss
        mse = self.mse_loss(y_pred, y_true)
        
        # Direction prediction loss
        # Calculate if the prediction correctly identifies the direction of movement
        pred_direction = torch.sign(y_pred[:, 0] - y_true[:, 0].roll(1))
        true_direction = torch.sign(y_true[:, 0] - y_true[:, 0].roll(1))
        direction_match = torch.mean((pred_direction == true_direction).float())
        
        # Combine losses - higher weight for correct direction
        total_loss = mse - self.direction_weight * direction_match
        
        return total_loss

# Learning rate scheduler
def get_lr_scheduler(optimizer):
    """Creates a learning rate scheduler with warm-up and decay"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=0.00001,
        verbose=True
    )

# Function to create an ensemble of models
def create_model_ensemble(input_dim, hidden_dims, num_layers_list, output_dim):
    """Create an ensemble of models with different architectures"""
    models = []
    
    for hidden_dim in hidden_dims:
        for num_layers in num_layers_list:
            # Standard bidirectional GRU with attention
            models.append(ImprovedGRUModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                bidirectional=True
            ))
            
            # Unidirectional GRU (for diversity)
            models.append(ImprovedGRUModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                bidirectional=False
            ))
    
    return models 