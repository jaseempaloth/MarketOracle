#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM Stock Price Prediction Model

This script trains and evaluates an LSTM model for stock price prediction
using the same data pipeline as the improved GRU model.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import argparse
from datetime import datetime, timedelta

# Import improved pipeline
from improved_pipeline import ImprovedDataPipeline
# Import model trainer
from model_trainer import ModelTrainer
# Import the custom LSTM model
from improved_lstm_model import ImprovedLSTMModel
# Import custom loss function from improved model
from improved_model import CustomLoss, get_lr_scheduler

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("lstm_training")

class LSTMTrainer:
    def __init__(self, device=None):
        self.pipeline = ImprovedDataPipeline()
        
        # Reuse the ModelTrainer for data preparation
        self.model_trainer = ModelTrainer(device)
        self.device = self.model_trainer.device
            
        logger.info(f"Using device: {self.device}")
    
    def create_lstm_model(self, input_dim, hidden_dim=128, num_layers=2, 
                         dropout=0.2, bidirectional=True):
        """Create an LSTM model with the same architecture as the GRU model"""
        model = ImprovedLSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        logger.info(f"Created LSTM model: {model}")
        return model
    
    def train_model(self, model, data_dict, epochs=100, batch_size=32,
                   learning_rate=0.001, patience=15, custom_loss=True):
        """Train the LSTM model using the same method as the GRU model"""
        # Extract tensors from data dictionary
        X_train_tensor = data_dict['X_train_tensor']
        y_train_tensor = data_dict['y_train_tensor']
        X_val_tensor = data_dict['X_val_tensor']
        y_val_tensor = data_dict['y_val_tensor']
        
        # Define loss function
        if custom_loss:
            criterion = CustomLoss(direction_weight=0.3)
        else:
            criterion = torch.nn.MSELoss()
        
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Add learning rate scheduler
        scheduler = get_lr_scheduler(optimizer)
        
        # Create data loader for batch training
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize tracking variables
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Train mode
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # Validation mode
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
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
                best_model_state = model.state_dict().copy()
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load the best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('LSTM - Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses)
        }
    
    def evaluate_model(self, model, data_dict):
        """Evaluate the LSTM model and calculate metrics"""
        # Just reuse the evaluation function from ModelTrainer
        return self.model_trainer.evaluate_model(model, data_dict)
    
    def compare_with_gru(self, lstm_result, gru_model, data_dict):
        """Compare LSTM performance with GRU on the same test set"""
        # Evaluate GRU model
        gru_metrics = self.model_trainer.evaluate_model(gru_model, data_dict)
        
        # Evaluate LSTM model
        lstm_metrics = self.evaluate_model(lstm_result['model'], data_dict)
        
        # Display comparison
        logger.info("\nLSTM vs GRU Performance Comparison:")
        logger.info(f"{'Metric':<20} {'LSTM':<15} {'GRU':<15}")
        logger.info("-" * 50)
        logger.info(f"{'RMSE':<20} ${lstm_metrics['rmse']:<14.2f} ${gru_metrics['rmse']:<14.2f}")
        logger.info(f"{'MAE':<20} ${lstm_metrics['mae']:<14.2f} ${gru_metrics['mae']:<14.2f}")
        logger.info(f"{'R² Score':<20} {lstm_metrics['r2']:<14.4f} {gru_metrics['r2']:<14.4f}")
        logger.info(f"{'Directional Acc.':<20} {lstm_metrics['directional_accuracy']:<14.2f}% {gru_metrics['directional_accuracy']:<14.2f}%")
        
        # Plot comparison of predictions
        val_dates = data_dict['val_dates']
        
        plt.figure(figsize=(14, 7))
        plt.plot(val_dates, lstm_metrics['y_val_inv'], label='Actual Prices', color='blue')
        plt.plot(val_dates, lstm_metrics['y_pred_inv'], label='LSTM Predictions', color='red', linestyle='--')
        plt.plot(val_dates, gru_metrics['y_pred_inv'], label='GRU Predictions', color='green', linestyle=':')
        plt.title('LSTM vs GRU Stock Price Prediction Comparison')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return {
            'lstm_metrics': lstm_metrics,
            'gru_metrics': gru_metrics
        }
    
    def save_model(self, model, path="improved_lstm_model.pth"):
        """Save the trained LSTM model to disk"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': model.lstm.input_size,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'output_dim': 1,
            'bidirectional': model.bidirectional
        }, path)
        
        logger.info(f"LSTM model saved to {path}")
    
    def load_model(self, path):
        """Load a trained LSTM model from disk"""
        checkpoint = torch.load(path)
        
        model = ImprovedLSTMModel(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            output_dim=checkpoint['output_dim'],
            bidirectional=checkpoint.get('bidirectional', True)
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"LSTM model loaded from {path}")
        return model


def run_lstm_comparison(ticker="AAPL", gru_model_path=None):
    """Train and evaluate LSTM model and compare with GRU"""
    # Initialize trainers
    lstm_trainer = LSTMTrainer()
    
    # Prepare data (reuse the data preparation from ModelTrainer)
    logger.info(f"Preparing data for {ticker}...")
    data = lstm_trainer.model_trainer.prepare_data(
        ticker=ticker, 
        period="5y",
        window_size=20,
        train_ratio=0.8,
        include_market_data=True
    )
    
    # Create LSTM model with the same architecture as GRU
    input_dim = data['X_train'].shape[2]
    lstm_model = lstm_trainer.create_lstm_model(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    )
    
    # Train LSTM model
    logger.info("Training LSTM model...")
    lstm_result = lstm_trainer.train_model(
        model=lstm_model,
        data_dict=data,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        patience=15,
        custom_loss=True
    )
    
    logger.info(f"LSTM model trained for {lstm_result['epochs_trained']} epochs")
    logger.info(f"Best LSTM validation loss: {lstm_result['best_val_loss']:.6f}")
    
    # Evaluate LSTM model
    logger.info("Evaluating LSTM model performance...")
    lstm_metrics = lstm_trainer.evaluate_model(lstm_result['model'], data)
    
    # Load or train GRU model for comparison
    if gru_model_path and os.path.exists(gru_model_path):
        # Load existing GRU model
        gru_model = lstm_trainer.model_trainer.load_model(gru_model_path)
        logger.info(f"Loaded GRU model from {gru_model_path} for comparison")
    else:
        # Train a new GRU model
        logger.info("No GRU model provided, training a new one for comparison...")
        gru_model = lstm_trainer.model_trainer.create_model(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2
        )
        gru_result = lstm_trainer.model_trainer.train_model(
            model=gru_model,
            data_dict=data
        )
        gru_model = gru_result['model']
    
    # Compare LSTM with GRU
    comparison = lstm_trainer.compare_with_gru(lstm_result, gru_model, data)
    
    # Save the LSTM model
    lstm_model_path = f"improved_{ticker}_lstm_model.pth"
    lstm_trainer.save_model(lstm_result['model'], lstm_model_path)
    
    logger.info("\nLSTM model training and evaluation completed successfully!")
    
    return lstm_result, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate LSTM stock price prediction model')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--gru-model', type=str, default=None, help='Path to pretrained GRU model')
    
    args = parser.parse_args()
    
    # Find existing GRU model if not specified
    if not args.gru_model:
        potential_path = f"improved_{args.ticker}_model.pth"
        if os.path.exists(potential_path):
            args.gru_model = potential_path
            print(f"Found existing GRU model at {potential_path}")
    
    run_lstm_comparison(args.ticker, args.gru_model) 