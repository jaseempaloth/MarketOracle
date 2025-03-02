import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import logging
import os
from datetime import datetime, timedelta
import traceback

from improved_model import ImprovedGRUModel, CustomLoss, get_lr_scheduler, create_model_ensemble
from improved_pipeline import ImprovedDataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_trainer")

class ModelTrainer:
    def __init__(self, device=None):
        self.pipeline = ImprovedDataPipeline()
        
        # Automatically select device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, ticker: str, period: str = "5y", window_size: int = 20, 
                   train_ratio: float = 0.8, include_market_data: bool = True) -> Dict:
        """Prepare data for training and evaluation"""
        try:
            # Fetch and preprocess data
            logger.info(f"Fetching data for {ticker}")
            stock_data = self.pipeline.fetch_stock_data(ticker, period)
            
            # Get market data if requested
            market_data = None
            if include_market_data:
                try:
                    market_data = self.pipeline.fetch_market_data(period)
                    logger.info(f"Successfully fetched market data with {len(market_data)} rows")
                except Exception as e:
                    logger.error(f"Failed to fetch market data: {str(e)}")
                    logger.info("Continuing without market data")
                    market_data = None
            
            # Try to preprocess data with error handling
            try:
                logger.info("Preprocessing data")
                scaled_data, original_data = self.pipeline.preprocess_data(stock_data, market_data)
            except Exception as e:
                logger.error(f"Error in standard preprocessing: {str(e)}")
                logger.info("Falling back to simplified preprocessing without market data")
                # Fall back to simplified preprocessing without market data
                scaled_data, original_data = self.pipeline.preprocess_data(stock_data, None)
            
            # Create sequences with step size of 1 day (overlapping windows for more training data)
            logger.info("Creating sequences")
            X, y = self.pipeline.create_sequences(scaled_data, window_size, step_size=1)
            
            # Split data
            logger.info("Splitting data into train and validation sets")
            X_train, X_val, y_train, y_val, split_idx = self.pipeline.split_data(X, y, train_ratio)
            
            # Convert to PyTorch tensors
            logger.info("Converting data to PyTorch tensors")
            X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = self.pipeline.prepare_tensors(
                X_train, X_val, y_train, y_val, self.device
            )
            
            # Get dates for plotting
            original_dates = original_data.index
            val_dates = original_dates[window_size+split_idx:window_size+split_idx+len(y_val)]
            
            logger.info(f"Data preparation complete: {len(X_train)} training samples, {len(X_val)} validation samples")
            
            return {
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val,
                'X_train_tensor': X_train_tensor,
                'X_val_tensor': X_val_tensor,
                'y_train_tensor': y_train_tensor,
                'y_val_tensor': y_val_tensor,
                'original_data': original_data,
                'scaled_data': scaled_data,
                'split_idx': split_idx,
                'window_size': window_size,
                'val_dates': val_dates
            }
        except Exception as e:
            logger.error(f"Failed to prepare data: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error preparing data: {str(e)}")
    
    def create_model(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                    dropout: float = 0.2, bidirectional: bool = True) -> ImprovedGRUModel:
        """Create an improved GRU model"""
        model = ImprovedGRUModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        logger.info(f"Created model: {model}")
        return model
    
    def train_model(self, model: ImprovedGRUModel, data_dict: Dict, 
                   epochs: int = 100, batch_size: int = 32, 
                   learning_rate: float = 0.001, patience: int = 15,
                   custom_loss: bool = True) -> Dict:
        """Train the model with early stopping"""
        # Extract tensors from data dictionary
        X_train_tensor = data_dict['X_train_tensor']
        y_train_tensor = data_dict['y_train_tensor']
        X_val_tensor = data_dict['X_val_tensor']
        y_val_tensor = data_dict['y_val_tensor']
        
        # Define loss function
        if custom_loss:
            criterion = CustomLoss(direction_weight=0.3)
        else:
            criterion = nn.MSELoss()
        
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
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
        plt.title('Training and Validation Loss')
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
    
    def evaluate_model(self, model: ImprovedGRUModel, data_dict: Dict) -> Dict:
        """Evaluate the model and calculate performance metrics"""
        # Extract data from dictionary
        X_val_tensor = data_dict['X_val_tensor']
        y_val = data_dict['y_val']
        original_data = data_dict['original_data']
        val_dates = data_dict['val_dates']
        
        # Set model to evaluation mode
        model.eval()
        
        # Make predictions
        with torch.no_grad():
            y_pred = model(X_val_tensor).cpu().numpy()
        
        # Inverse transform predictions and actual values
        y_val_inv = np.array([self.pipeline.inverse_transform_price(val, original_data) for val in y_val])
        y_pred_inv = np.array([self.pipeline.inverse_transform_price(val[0], original_data) for val in y_pred])
        
        # Calculate metrics
        mse = mean_squared_error(y_val_inv, y_pred_inv)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_val_inv, y_pred_inv)
        r2 = r2_score(y_val_inv, y_pred_inv)
        
        # Calculate directional accuracy (percentage of correct direction predictions)
        actual_dirs = np.sign(np.diff(np.append([y_val_inv[0]], y_val_inv)))
        pred_dirs = np.sign(np.diff(np.append([y_val_inv[0]], y_pred_inv)))
        dir_accuracy = np.mean(actual_dirs == pred_dirs) * 100
        
        # Display metrics
        logger.info(f'Mean Squared Error: {mse:.4f}')
        logger.info(f'Root Mean Squared Error: {rmse:.4f}')
        logger.info(f'Mean Absolute Error: {mae:.4f}')
        logger.info(f'R² Score: {r2:.4f}')
        logger.info(f'Directional Accuracy: {dir_accuracy:.2f}%')
        
        # Visualize predictions vs actual values
        plt.figure(figsize=(14, 7))
        plt.plot(val_dates, y_val_inv, label='Actual Prices', color='blue')
        plt.plot(val_dates, y_pred_inv, label='Predicted Prices', color='red', linestyle='--')
        plt.title('Stock Price Prediction vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': dir_accuracy,
            'y_val_inv': y_val_inv,
            'y_pred_inv': y_pred_inv,
            'evaluation_dates': val_dates
        }
    
    def forecast_future(self, model: ImprovedGRUModel, data_dict: Dict, days: int = 20) -> Dict:
        """Generate forecasts for future days"""
        # Extract required data
        scaled_data = data_dict['scaled_data']
        original_data = data_dict['original_data']
        window_size = data_dict['window_size']
        
        # Set model to evaluation mode
        model.eval()
        
        # Get the most recent data window
        last_window = scaled_data[-window_size:].reshape(1, window_size, -1)
        current_input = torch.tensor(last_window, dtype=torch.float32).to(self.device)
        
        # Forecast dates
        last_date = original_data.index[-1]
        forecast_dates = [(last_date + timedelta(days=i+1)) for i in range(days)]
        
        # Initialize predictions list
        predictions = []
        
        # Make predictions for each day
        with torch.no_grad():
            for i in range(days):
                # Get prediction for the next day
                pred = model(current_input).cpu().numpy()[0, 0]
                predictions.append(pred)
                
                # Create a new row with the same values as the last row
                new_pred_row = current_input[0, -1, :].clone().cpu().numpy()
                
                # Update the Close price
                new_pred_row[self.pipeline.close_idx] = pred
                
                # Update current_input by removing the first row and adding the new prediction
                current_input = torch.roll(current_input, -1, dims=1)
                current_input[0, -1, :] = torch.tensor(new_pred_row, dtype=torch.float32).to(self.device)
        
        # Convert scaled predictions back to original scale
        original_predictions = [self.pipeline.inverse_transform_price(pred, original_data) for pred in predictions]
        
        # Plot the forecast
        plt.figure(figsize=(14, 7))
        # Plot historical data (last 60 days)
        plt.plot(original_data.index[-60:], original_data['Close'][-60:], label='Historical Prices', color='blue')
        
        # Plot forecasted data
        plt.plot(forecast_dates, original_predictions, label='Forecasted Prices', color='red', marker='o')
        plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
        plt.title(f'Stock Price Forecast for Next {days} Days')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Print forecasted values
        logger.info("\nForecasted Prices:")
        for date, price in zip(forecast_dates, original_predictions):
            logger.info(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
        
        return {
            'forecast_dates': forecast_dates,
            'forecast_prices': original_predictions
        }
    
    def train_ensemble(self, data_dict: Dict, epochs: int = 50) -> Dict:
        """Train an ensemble of models with different architectures"""
        # Extract dimensions from data
        input_dim = data_dict['X_train'].shape[2]
        
        # Define model variations
        hidden_dims = [64, 128, 192]
        num_layers_list = [1, 2]
        
        # Create ensemble
        models = create_model_ensemble(input_dim, hidden_dims, num_layers_list, output_dim=1)
        logger.info(f"Created ensemble with {len(models)} models")
        
        # Train each model with reduced epochs
        trained_models = []
        for i, model in enumerate(models):
            logger.info(f"Training ensemble model {i+1}/{len(models)}")
            model = model.to(self.device)
            result = self.train_model(
                model, data_dict, 
                epochs=epochs,  # Reduced epochs for ensemble
                batch_size=32,
                learning_rate=0.001,
                patience=10  # Reduced patience for ensemble
            )
            trained_models.append(result['model'])
        
        # Combine the models into an ensemble
        ensemble = {
            'models': trained_models,
            'count': len(trained_models)
        }
        
        return ensemble
    
    def predict_with_ensemble(self, ensemble: Dict, X_tensor: torch.Tensor) -> np.ndarray:
        """Make predictions using an ensemble of models"""
        predictions = []
        
        # Get predictions from each model
        for model in ensemble['models']:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy()
                predictions.append(pred)
        
        # Average the predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def evaluate_ensemble(self, ensemble: Dict, data_dict: Dict) -> Dict:
        """Evaluate the ensemble model"""
        # Make predictions with the ensemble
        ensemble_pred = self.predict_with_ensemble(ensemble, data_dict['X_val_tensor'])
        
        # Prepare for metric calculation
        y_val = data_dict['y_val']
        original_data = data_dict['original_data']
        val_dates = data_dict['val_dates']
        
        # Inverse transform predictions and actual values
        y_val_inv = np.array([self.pipeline.inverse_transform_price(val, original_data) for val in y_val])
        y_pred_inv = np.array([self.pipeline.inverse_transform_price(val[0], original_data) for val in ensemble_pred])
        
        # Calculate metrics
        mse = mean_squared_error(y_val_inv, y_pred_inv)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_val_inv, y_pred_inv)
        r2 = r2_score(y_val_inv, y_pred_inv)
        
        # Calculate directional accuracy
        actual_dirs = np.sign(np.diff(np.append([y_val_inv[0]], y_val_inv)))
        pred_dirs = np.sign(np.diff(np.append([y_val_inv[0]], y_pred_inv)))
        dir_accuracy = np.mean(actual_dirs == pred_dirs) * 100
        
        # Display metrics
        logger.info(f'Ensemble Metrics:')
        logger.info(f'Mean Squared Error: {mse:.4f}')
        logger.info(f'Root Mean Squared Error: {rmse:.4f}')
        logger.info(f'Mean Absolute Error: {mae:.4f}')
        logger.info(f'R² Score: {r2:.4f}')
        logger.info(f'Directional Accuracy: {dir_accuracy:.2f}%')
        
        # Visualize predictions vs actual values
        plt.figure(figsize=(14, 7))
        plt.plot(val_dates, y_val_inv, label='Actual Prices', color='blue')
        plt.plot(val_dates, y_pred_inv, label='Ensemble Predictions', color='red', linestyle='--')
        plt.title('Ensemble Stock Price Prediction vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': dir_accuracy,
            'y_val_inv': y_val_inv,
            'y_pred_inv': y_pred_inv,
            'evaluation_dates': val_dates
        }
    
    def save_model(self, model: ImprovedGRUModel, path: str = "improved_model.pth"):
        """Save the trained model to disk"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': model.gru.input_size,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'output_dim': 1,
            'bidirectional': model.bidirectional
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def save_ensemble(self, ensemble: Dict, base_path: str = "ensemble_model"):
        """Save all models in the ensemble"""
        os.makedirs(base_path, exist_ok=True)
        for i, model in enumerate(ensemble['models']):
            path = os.path.join(base_path, f"model_{i+1}.pth")
            self.save_model(model, path)
        
        logger.info(f"Ensemble with {len(ensemble['models'])} models saved to {base_path}/")
    
    def load_model(self, path: str) -> ImprovedGRUModel:
        """Load a trained model from disk"""
        checkpoint = torch.load(path)
        
        model = ImprovedGRUModel(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            output_dim=checkpoint['output_dim'],
            bidirectional=checkpoint.get('bidirectional', True)
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def load_ensemble(self, base_path: str) -> Dict:
        """Load an ensemble of models from disk"""
        model_files = [f for f in os.listdir(base_path) if f.endswith('.pth')]
        
        models = []
        for file in model_files:
            path = os.path.join(base_path, file)
            model = self.load_model(path)
            models.append(model)
        
        logger.info(f"Loaded ensemble with {len(models)} models from {base_path}/")
        
        return {
            'models': models,
            'count': len(models)
        }


# Example of how to use:
if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data for Apple stock
    data = trainer.prepare_data(ticker="AAPL", period="5y", window_size=20)
    
    # Create model with improved architecture
    input_dim = data['X_train'].shape[2]
    model = trainer.create_model(input_dim=input_dim, hidden_dim=128, num_layers=2)
    
    # Train model
    training_result = trainer.train_model(model, data, epochs=100, batch_size=32, 
                                         learning_rate=0.001, custom_loss=True)
    
    # Evaluate model
    eval_result = trainer.evaluate_model(training_result['model'], data)
    
    # Generate forecast
    forecast = trainer.forecast_future(training_result['model'], data, days=20)
    
    # Save model
    trainer.save_model(training_result['model'], "improved_stock_model.pth") 