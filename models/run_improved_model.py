
"""
Improved Stock Price Prediction Model Demo

This script demonstrates how to use the improved stock price prediction model
with bidirectional GRU, attention mechanism, enhanced features, and ensemble methods.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

# Import our improved modules
from improved_model import ImprovedGRUModel, CustomLoss, get_lr_scheduler, create_model_ensemble
from improved_pipeline import ImprovedDataPipeline
from model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("stock_prediction_demo")

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available 
              else 'seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)

def run_demo():
    """Run the improved stock price prediction model demo"""
    # Choose a stock ticker
    ticker = "AAPL"  # Apple Inc.
    
    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize the trainer
    trainer = ModelTrainer(device=device)
    logger.info("Model trainer initialized successfully.")
    
    # Prepare data with enhanced features
    logger.info(f"Preparing data for {ticker}...")
    data = trainer.prepare_data(
        ticker=ticker, 
        period="5y",  # Longer period for more training data
        window_size=20,  # 20 trading days (about a month) to predict next day
        train_ratio=0.8,  # 80% train, 20% validation
        include_market_data=True  # Include S&P 500 and VIX data
    )
    
    logger.info(f"Dataset prepared for {ticker}")
    logger.info(f"Training samples: {len(data['X_train'])}")
    logger.info(f"Validation samples: {len(data['X_val'])}")
    logger.info(f"Number of features: {data['X_train'].shape[2]}")
    
    # Create the improved model
    input_dim = data['X_train'].shape[2]  # Number of features
    model = trainer.create_model(
        input_dim=input_dim,
        hidden_dim=128,  # Larger hidden dimension for more capacity
        num_layers=2,    # Multiple layers for hierarchical feature learning
        dropout=0.2,     # Regularization to prevent overfitting
        bidirectional=True  # Bidirectional to capture patterns in both directions
    )
    
    # Train the model
    logger.info("Training improved model...")
    training_result = trainer.train_model(
        model=model,
        data_dict=data,
        epochs=100,       # Maximum number of epochs
        batch_size=32,    # Batch size for mini-batch training
        learning_rate=0.001,  # Initial learning rate
        patience=15,      # Early stopping patience
        custom_loss=True  # Use our custom loss function that considers price direction
    )
    
    logger.info(f"Model trained for {training_result['epochs_trained']} epochs")
    logger.info(f"Best validation loss: {training_result['best_val_loss']:.6f}")
    
    # Evaluate the model
    logger.info("Evaluating model performance...")
    eval_result = trainer.evaluate_model(training_result['model'], data)
    
    # Print summary of metrics
    logger.info("\nModel Performance Summary:")
    logger.info(f"Root Mean Squared Error (RMSE): {eval_result['rmse']:.2f}")
    logger.info(f"Mean Absolute Error (MAE): {eval_result['mae']:.2f}")
    logger.info(f"R² Score: {eval_result['r2']:.4f}")
    logger.info(f"Directional Accuracy: {eval_result['directional_accuracy']:.2f}%")
    
    # Train an ensemble (optional, can be time-consuming)
    train_ensemble = False  # Set to True to train ensemble
    
    if train_ensemble:
        logger.info("Training model ensemble (this may take some time)...")
        ensemble = trainer.train_ensemble(data, epochs=50)
        
        logger.info(f"Trained ensemble with {ensemble['count']} different model architectures")
        
        # Evaluate the ensemble
        logger.info("Evaluating ensemble performance...")
        ensemble_result = trainer.evaluate_ensemble(ensemble, data)
        
        # Compare with single model
        logger.info("\nPerformance Comparison:")
        logger.info("Metric\t\tSingle Model\tEnsemble")
        logger.info(f"RMSE:\t\t{eval_result['rmse']:.2f}\t\t{ensemble_result['rmse']:.2f}")
        logger.info(f"MAE:\t\t{eval_result['mae']:.2f}\t\t{ensemble_result['mae']:.2f}")
        logger.info(f"R²:\t\t{eval_result['r2']:.4f}\t\t{ensemble_result['r2']:.4f}")
        logger.info(f"Direction:\t{eval_result['directional_accuracy']:.2f}%\t\t{ensemble_result['directional_accuracy']:.2f}%")
        
        # Determine which model to use for forecasting
        use_ensemble = ensemble_result['rmse'] < eval_result['rmse']
    else:
        use_ensemble = False
        ensemble = None
    
    # Generate forecast for the next 30 days
    logger.info("Generating price forecast...")
    forecast_days = 30
    
    if use_ensemble and ensemble is not None:
        # Use the first model from ensemble for simplicity
        forecast = trainer.forecast_future(ensemble['models'][0], data, days=forecast_days)
    else:
        forecast = trainer.forecast_future(training_result['model'], data, days=forecast_days)
    
    # Save the model
    logger.info("Saving model for future use...")
    if use_ensemble and ensemble is not None:
        # Save the entire ensemble
        trainer.save_ensemble(ensemble, base_path="ensemble_models")
        logger.info("Ensemble model saved to 'ensemble_models/' directory")
    else:
        # Save single model
        trainer.save_model(training_result['model'], f"improved_{ticker}_model.pth")
        logger.info(f"Model saved to 'improved_{ticker}_model.pth'")
    
    logger.info("\nDemo completed successfully!")
    logger.info(f"Model RMSE: {eval_result['rmse']:.2f}")
    logger.info(f"Model directional accuracy: {eval_result['directional_accuracy']:.2f}%")

if __name__ == "__main__":
    run_demo() 