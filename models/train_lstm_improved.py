#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved LSTM Stock Price Prediction Model

This script trains an LSTM model with optimized parameters for stock price prediction.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
import argparse

# Import from existing files
from improved_pipeline import ImprovedDataPipeline
from model_trainer import ModelTrainer
from improved_lstm_model import ImprovedLSTMModel
from improved_model import CustomLoss, get_lr_scheduler

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("lstm_improved")

def train_optimized_lstm(ticker="AAPL"):
    """Train an LSTM model with optimized parameters"""
    # Initialize model trainer
    model_trainer = ModelTrainer()
    device = model_trainer.device
    
    logger.info(f"Training optimized LSTM model for {ticker}...")
    
    # Prepare data with the same pipeline
    data = model_trainer.prepare_data(
        ticker=ticker, 
        period="5y",
        window_size=20,
        train_ratio=0.8,
        include_market_data=True
    )
    
    # Get the pipeline instance that was used to process the data (important!)
    pipeline = model_trainer.pipeline
    
    # Create the LSTM model with improved parameters
    input_dim = data['X_train'].shape[2]
    
    # Define optimized LSTM model with larger capacity
    lstm_model = ImprovedLSTMModel(
        input_dim=input_dim,
        hidden_dim=256,  # Increased from 128
        num_layers=3,    # Increased from 2
        output_dim=1,
        dropout=0.3,     # Increased from 0.2
        bidirectional=True
    ).to(device)
    
    logger.info(f"Created optimized LSTM model: {lstm_model}")
    
    # Define optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(
        lstm_model.parameters(), 
        lr=0.0005,  # Lower learning rate
        weight_decay=1e-5  # L2 regularization
    )
    
    # Define custom loss with higher direction weight
    criterion = CustomLoss(direction_weight=0.5)  # Increased from 0.3
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=7, 
        min_lr=0.00001,
        verbose=True
    )
    
    # Create data loader with smaller batch size
    X_train_tensor = data['X_train_tensor']
    y_train_tensor = data['y_train_tensor']
    X_val_tensor = data['X_val_tensor']
    y_val_tensor = data['y_val_tensor']
    
    batch_size = 16  # Smaller batch size for better generalization
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training parameters
    epochs = 150  # Increased from 100
    patience = 20  # Increased from 15
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    
    # Training loop
    logger.info("Starting LSTM training with optimized parameters...")
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
            
            # Gradient clipping to prevent exploding gradients (lower threshold)
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=0.5)
            
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
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch+1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = lstm_model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load the best model
    if best_model_state is not None:
        lstm_model.load_state_dict(best_model_state)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Optimized LSTM - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    loss_plot_path = f'lstm_training_loss_{ticker}.png'
    plt.savefig(loss_plot_path)
    logger.info(f"Training loss plot saved to {loss_plot_path}")
    
    # Evaluate the model
    lstm_model.eval()
    with torch.no_grad():
        y_pred = lstm_model(X_val_tensor).cpu().numpy()
    
    # Inverse transform predictions and actual values
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
    logger.info("\nOptimized LSTM Performance:")
    logger.info(f"Root Mean Squared Error: ${rmse:.2f}")
    logger.info(f"Mean Absolute Error: ${mae:.2f}")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"Directional Accuracy: {dir_accuracy:.2f}%")
    
    # Plot predictions vs actual
    plt.figure(figsize=(14, 7))
    plt.plot(val_dates, y_val_inv, label='Actual Prices', color='blue')
    plt.plot(val_dates, y_pred_inv, label='LSTM Predictions', color='red', linestyle='--')
    plt.title(f'Optimized LSTM Stock Price Prediction for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the prediction plot
    pred_plot_path = f'lstm_predictions_{ticker}.png'
    plt.savefig(pred_plot_path)
    logger.info(f"Prediction plot saved to {pred_plot_path}")
    
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
        
        # Calculate GRU metrics
        gru_mse = mean_squared_error(y_val_inv, gru_pred_inv)
        gru_rmse = math.sqrt(gru_mse)
        gru_mae = mean_absolute_error(y_val_inv, gru_pred_inv)
        gru_r2 = r2_score(y_val_inv, gru_pred_inv)
        
        # Calculate GRU directional accuracy
        gru_dirs = np.sign(np.diff(np.append([y_val_inv[0]], gru_pred_inv)))
        gru_dir_accuracy = np.mean(actual_dirs == gru_dirs) * 100
        
        # Compare LSTM with GRU
        logger.info("\nLSTM vs GRU Performance Comparison:")
        logger.info(f"{'Metric':<20} {'Optimized LSTM':<15} {'GRU':<15}")
        logger.info("-" * 50)
        logger.info(f"{'RMSE':<20} ${rmse:<14.2f} ${gru_rmse:<14.2f}")
        logger.info(f"{'MAE':<20} ${mae:<14.2f} ${gru_mae:<14.2f}")
        logger.info(f"{'R² Score':<20} {r2:<14.4f} {gru_r2:<14.4f}")
        logger.info(f"{'Directional Acc.':<20} {dir_accuracy:<14.2f}% {gru_dir_accuracy:<14.2f}%")
        
        # Plot comparison
        plt.figure(figsize=(14, 7))
        plt.plot(val_dates, y_val_inv, label='Actual Prices', color='blue')
        plt.plot(val_dates, y_pred_inv, label='Optimized LSTM', color='red', linestyle='--')
        plt.plot(val_dates, gru_pred_inv, label='GRU', color='green', linestyle=':')
        plt.title('LSTM vs GRU Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the comparison plot
        comp_plot_path = f'lstm_vs_gru_{ticker}.png'
        plt.savefig(comp_plot_path)
        logger.info(f"Model comparison plot saved to {comp_plot_path}")
    
    # Save the optimized LSTM model
    lstm_model_path = f"improved_{ticker}_optimized_lstm_model.pth"
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'input_dim': lstm_model.lstm.input_size,
        'hidden_dim': lstm_model.hidden_dim,
        'num_layers': lstm_model.num_layers,
        'output_dim': 1,
        'bidirectional': lstm_model.bidirectional
    }, lstm_model_path)
    
    logger.info(f"Optimized LSTM model saved to {lstm_model_path}")
    
    return {
        'model': lstm_model,
        'lstm_metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': dir_accuracy
        },
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_trained': len(train_losses)
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train an optimized LSTM model for stock price prediction')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    args = parser.parse_args()
    
    # Train the model
    train_optimized_lstm(args.ticker) 