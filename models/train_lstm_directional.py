#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Directional LSTM Model for Stock Price Prediction

This script trains an LSTM model specifically optimized for directional prediction
rather than absolute price values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import argparse
from datetime import datetime

# Import from existing files
from improved_pipeline import ImprovedDataPipeline
from model_trainer import ModelTrainer
from improved_lstm_model import ImprovedLSTMModel

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("lstm_directional")

class DirectionalLoss(nn.Module):
    """
    Custom loss function that prioritizes directional accuracy over absolute price differences.
    
    This loss combines MSE (for basic convergence) with a directional component that penalizes
    incorrect direction predictions more heavily.
    """
    def __init__(self, direction_weight=0.8):
        super(DirectionalLoss, self).__init__()
        self.direction_weight = direction_weight
        self.mse_weight = 1.0 - direction_weight
        
    def forward(self, y_pred, y_true):
        # Basic MSE component for convergence
        mse_loss = F.mse_loss(y_pred, y_true)
        
        # For directional component, we need to create shifted versions
        # Get the batch size
        batch_size = y_pred.shape[0]
        
        # If batch size is less than 2, we can't calculate direction
        if batch_size < 2:
            return mse_loss
        
        # Calculate price changes for predictions and true values
        # For each sample except the first, compare with the previous sample
        y_pred_diff = y_pred[1:] - y_pred[:-1]
        y_true_diff = y_true[1:] - y_true[:-1]
        
        # Get the signs of the differences
        y_pred_sign = torch.sign(y_pred_diff)
        y_true_sign = torch.sign(y_true_diff)
        
        # Calculate directional loss - penalize when signs don't match
        # Convert matching signs to 1.0 (correct) and non-matching to -1.0 (incorrect)
        sign_match = y_pred_sign * y_true_sign
        
        # Convert to a loss (1.0 for correct predictions, 0.0 for incorrect)
        # We add 1 and divide by 2 to map from [-1, 1] to [0, 1]
        directional_accuracy = (sign_match + 1) / 2
        
        # Convert to a loss (higher for incorrect predictions)
        directional_loss = 1 - directional_accuracy.mean()
        
        # Combine the losses with their respective weights
        combined_loss = (self.mse_weight * mse_loss) + (self.direction_weight * directional_loss)
        
        return combined_loss

def train_directional_lstm(ticker="AAPL"):
    """Train an LSTM model optimized for directional prediction"""
    # Initialize model trainer
    model_trainer = ModelTrainer()
    device = model_trainer.device
    
    logger.info(f"Training directional LSTM model for {ticker}...")
    
    # Prepare data with the same pipeline
    data = model_trainer.prepare_data(
        ticker=ticker, 
        period="5y",
        window_size=20,
        train_ratio=0.8,
        include_market_data=True
    )
    
    # Get the pipeline instance that was used to process the data
    pipeline = model_trainer.pipeline
    
    # Create the LSTM model (smaller than the previous one to prevent overfitting)
    input_dim = data['X_train'].shape[2]
    
    # Define directional LSTM model
    lstm_model = ImprovedLSTMModel(
        input_dim=input_dim,
        hidden_dim=128,  # Smaller than before
        num_layers=2,    # Reduced layers
        output_dim=1,
        dropout=0.4,     # Higher dropout to prevent overfitting to exact values
        bidirectional=True
    ).to(device)
    
    logger.info(f"Created directional LSTM model: {lstm_model}")
    
    # Define optimizer with higher weight decay for stronger regularization
    optimizer = torch.optim.Adam(
        lstm_model.parameters(), 
        lr=0.001,       # Higher learning rate for directional tasks
        weight_decay=1e-4  # Stronger L2 regularization
    )
    
    # Define custom directional loss with high direction weight
    criterion = DirectionalLoss(direction_weight=0.8)  # Heavy emphasis on direction
    
    # Create data loader
    X_train_tensor = data['X_train_tensor']
    y_train_tensor = data['y_train_tensor']
    X_val_tensor = data['X_val_tensor']
    y_val_tensor = data['y_val_tensor']
    
    batch_size = 32  # Larger batch size for better direction pattern recognition
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training parameters
    epochs = 200  # More epochs since we have a specific goal
    patience = 30  # Higher patience to allow finding directional patterns
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    val_dir_accuracies = []
    best_val_dir_acc = 0
    counter = 0
    best_model_state = None
    
    # Training loop
    logger.info("Starting directional LSTM training...")
    for epoch in range(epochs):
        # Train mode
        lstm_model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation mode
        lstm_model.eval()
        with torch.no_grad():
            val_outputs = lstm_model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_losses.append(val_loss)
            
            # Calculate directional accuracy on validation set
            val_pred_np = val_outputs.cpu().numpy().flatten()
            val_true_np = y_val_tensor.cpu().numpy().flatten()
            
            # For direction, we need the signs of consecutive differences
            val_pred_diff = np.diff(val_pred_np)
            val_true_diff = np.diff(val_true_np)
            
            val_pred_sign = np.sign(val_pred_diff)
            val_true_sign = np.sign(val_true_diff)
            
            # Directional accuracy
            matches = (val_pred_sign == val_true_sign)
            dir_accuracy = np.mean(matches) * 100  # as percentage
            val_dir_accuracies.append(dir_accuracy)
        
        # Print progress
        if (epoch+1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}, Dir Acc: {dir_accuracy:.2f}%')
        
        # Early stopping based on directional accuracy (not loss)
        if dir_accuracy > best_val_dir_acc:
            best_val_dir_acc = dir_accuracy
            counter = 0
            best_model_state = lstm_model.state_dict().copy()
            logger.info(f"New best directional accuracy: {best_val_dir_acc:.2f}%")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1} - Best dir acc: {best_val_dir_acc:.2f}%')
                break
    
    # Load the best model
    if best_model_state is not None:
        lstm_model.load_state_dict(best_model_state)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Loss curves
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Directional LSTM - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Directional accuracy
    plt.subplot(2, 1, 2)
    plt.plot(val_dir_accuracies, label='Directional Accuracy', color='green')
    plt.axhline(y=50, color='r', linestyle='--', label='Random Guess (50%)')
    plt.title('Directional LSTM - Validation Directional Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Directional Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    training_plot_path = f'lstm_directional_training_{ticker}.png'
    plt.savefig(training_plot_path)
    logger.info(f"Training metrics plot saved to {training_plot_path}")
    
    # Evaluate the model
    lstm_model.eval()
    with torch.no_grad():
        y_pred = lstm_model(X_val_tensor).cpu().numpy()
    
    # Get inverse transformed data for visualization
    y_val = data['y_val']
    original_data = data['original_data']
    val_dates = data['val_dates']
    
    # Check if pipeline.feature_columns is None before using inverse_transform_price
    if pipeline.feature_columns is None:
        logger.error("Pipeline feature_columns is None. Cannot inverse transform.")
        logger.info("Creating manual inverse transformation based on original data scaling")
        
        # Calculate the scaling factors from original data
        close_series = original_data['Close']
        min_close = close_series.min()
        max_close = close_series.max()
        
        # Define a simple manual inverse transformation function
        def manual_inverse_transform(scaled_val):
            return scaled_val * (max_close - min_close) + min_close
        
        # Apply manual transformation
        y_val_inv = np.array([manual_inverse_transform(val) for val in y_val])
        y_pred_inv = np.array([manual_inverse_transform(val[0]) for val in y_pred])
    else:
        # Use the pipeline's inverse transform if feature_columns is available
        y_val_inv = np.array([pipeline.inverse_transform_price(val, original_data) for val in y_val])
        y_pred_inv = np.array([pipeline.inverse_transform_price(val[0], original_data) for val in y_pred])
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import math
    
    mse = mean_squared_error(y_val_inv, y_pred_inv)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_val_inv, y_pred_inv)
    r2 = r2_score(y_val_inv, y_pred_inv)
    
    # Calculate directional accuracy
    actual_dirs = np.sign(np.diff(np.append([y_val_inv[0]], y_val_inv)))
    pred_dirs = np.sign(np.diff(np.append([y_val_inv[0]], y_pred_inv)))
    dir_accuracy = np.mean(actual_dirs == pred_dirs) * 100
    
    # Display metrics
    logger.info("\nDirectional LSTM Performance:")
    logger.info(f"Root Mean Squared Error: ${rmse:.2f}")
    logger.info(f"Mean Absolute Error: ${mae:.2f}")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"Directional Accuracy: {dir_accuracy:.2f}%")
    
    # Create a DataFrame for directional prediction results
    results_df = pd.DataFrame({
        'Date': val_dates,
        'Actual_Price': y_val_inv,
        'Predicted_Price': y_pred_inv
    })
    
    # Add columns for actual and predicted directions
    results_df['Actual_Direction'] = np.append([0], np.sign(np.diff(results_df['Actual_Price'])))
    results_df['Predicted_Direction'] = np.append([0], np.sign(np.diff(results_df['Predicted_Price'])))
    results_df['Direction_Match'] = results_df['Actual_Direction'] == results_df['Predicted_Direction']
    
    # Map directions to text for better readability
    direction_map = {1: 'Up', -1: 'Down', 0: 'Neutral'}
    results_df['Actual_Direction_Text'] = results_df['Actual_Direction'].map(direction_map)
    results_df['Predicted_Direction_Text'] = results_df['Predicted_Direction'].map(direction_map)
    
    # Save the results to CSV
    results_path = f'lstm_directional_results_{ticker}.csv'
    results_df.to_csv(results_path)
    logger.info(f"Detailed prediction results saved to {results_path}")
    
    # Create visualization of directional predictions
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Price predictions vs actual
    plt.subplot(2, 1, 1)
    plt.plot(val_dates, y_val_inv, label='Actual Prices', color='blue')
    plt.plot(val_dates, y_pred_inv, label='LSTM Predictions', color='red', linestyle='--')
    plt.title(f'Directional LSTM Price Predictions for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Direction visualization
    plt.subplot(2, 1, 2)
    
    # Create directional markers - green for correct predictions, red for incorrect
    correct_idx = results_df.index[results_df['Direction_Match'] & (results_df['Actual_Direction'] != 0)]
    incorrect_idx = results_df.index[(~results_df['Direction_Match']) & (results_df['Actual_Direction'] != 0)]
    
    # Plot actual price
    plt.plot(val_dates, y_val_inv, label='Actual Price', color='blue')
    
    # Plot correct and incorrect directional predictions
    if len(correct_idx) > 0:
        plt.scatter(
            [val_dates[i-results_df.index[0]] for i in correct_idx], 
            [y_val_inv[i-results_df.index[0]] for i in correct_idx],
            color='green', marker='^', s=50, label='Correct Direction'
        )
    
    if len(incorrect_idx) > 0:
        plt.scatter(
            [val_dates[i-results_df.index[0]] for i in incorrect_idx], 
            [y_val_inv[i-results_df.index[0]] for i in incorrect_idx],
            color='red', marker='v', s=50, label='Incorrect Direction'
        )
    
    plt.title(f'Directional Prediction Accuracy: {dir_accuracy:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the direction visualization
    direction_plot_path = f'lstm_direction_viz_{ticker}.png'
    plt.savefig(direction_plot_path)
    logger.info(f"Direction visualization saved to {direction_plot_path}")
    
    # Load the GRU model for comparison
    gru_model_path = f"improved_{ticker}_model.pth"
    if os.path.exists(gru_model_path):
        # Load GRU model
        gru_model = model_trainer.load_model(gru_model_path)
        logger.info(f"Loaded GRU model from {gru_model_path} for comparison")
        
        # Evaluate GRU model
        gru_model.eval()
        with torch.no_grad():
            gru_pred = gru_model(X_val_tensor).cpu().numpy()
        
        # Transform GRU predictions
        if pipeline.feature_columns is None:
            gru_pred_inv = np.array([manual_inverse_transform(val[0]) for val in gru_pred])
        else:
            gru_pred_inv = np.array([pipeline.inverse_transform_price(val[0], original_data) for val in gru_pred])
        
        # Calculate GRU directional accuracy
        gru_dirs = np.sign(np.diff(np.append([y_val_inv[0]], gru_pred_inv)))
        gru_dir_accuracy = np.mean(actual_dirs == gru_dirs) * 100
        
        # Compare directional accuracy with GRU
        logger.info("\nDirectional LSTM vs GRU Comparison:")
        logger.info(f"{'Metric':<20} {'Directional LSTM':<20} {'GRU':<15}")
        logger.info("-" * 55)
        logger.info(f"{'Directional Acc.':<20} {dir_accuracy:<19.2f}% {gru_dir_accuracy:<14.2f}%")
        
        # If we want, we could add other metrics here as well
        logger.info(f"{'RMSE':<20} ${rmse:<19.2f} ${math.sqrt(mean_squared_error(y_val_inv, gru_pred_inv)):<14.2f}")
        logger.info(f"{'R² Score':<20} {r2:<19.4f} {r2_score(y_val_inv, gru_pred_inv):<14.4f}")
    
    # Save the directional LSTM model
    lstm_model_path = f"improved_{ticker}_directional_lstm_model.pth"
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'input_dim': lstm_model.lstm.input_size,
        'hidden_dim': lstm_model.hidden_dim,
        'num_layers': lstm_model.num_layers,
        'output_dim': 1,
        'bidirectional': lstm_model.bidirectional,
        'directional_accuracy': dir_accuracy
    }, lstm_model_path)
    
    logger.info(f"Directional LSTM model saved to {lstm_model_path}")
    
    return {
        'model': lstm_model,
        'directional_accuracy': dir_accuracy,
        'rmse': rmse,
        'r2': r2,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_dir_accuracies': val_dir_accuracies
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a directional LSTM model for stock price prediction')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    args = parser.parse_args()
    
    # Train the model
    train_directional_lstm(args.ticker) 