import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("stock_model")

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class DataPipeline:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        
    def fetch_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetch historical stock data"""
        try:
            logger.info(f"Fetching data for {ticker} with period {period}")
            stock_data = yf.download(ticker, period=period)
            if stock_data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            logger.info(f"Successfully fetched {len(stock_data)} rows for {ticker}")
            return stock_data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for feature engineering"""
        try:
            logger.info("Calculating technical indicators")
            df = data.copy()
            
            # Simple Moving Averages (SMA)
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # Handle division by zero
            avg_loss = avg_loss.replace(0, np.nan)
            rs = avg_gain / avg_loss
            rs = rs.fillna(0)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            logger.info("Technical indicators calculated successfully")
            return df
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error calculating technical indicators: {str(e)}")
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Preprocess the stock data with technical indicators"""
        try:
            logger.info("Preprocessing data")
            # Calculate technical indicators
            data_with_indicators = self.calculate_technical_indicators(data)
            
            # Handle missing values created by indicators that use lookback periods
            data_with_indicators = data_with_indicators.bfill().ffill()
            
            # Select features for the model
            feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_10', 'SMA_50', 'RSI'
            ]
            
            # Ensure all selected features exist
            available_features = [col for col in feature_columns if col in data_with_indicators.columns]
            logger.info(f"Using features: {available_features}")
            
            # Store feature columns for later use
            self.feature_columns = available_features
            
            # Get the processed data with selected features
            processed_data = data_with_indicators[available_features].copy()
            
            # Store the column indices for reference
            self.column_indices = {col: i for i, col in enumerate(processed_data.columns)}
            self.close_idx = self.column_indices.get('Close', 3)  # Default to 3 if not found
            logger.info(f"Close price index: {self.close_idx}")
            
            # Feature scaling
            scaled_data = self.scaler.fit_transform(processed_data)
            logger.info("Data preprocessing completed successfully")
            
            return scaled_data, processed_data
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error preprocessing data: {str(e)}")
    
    def create_sequences(self, data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences for the model"""
        try:
            X, y = [], []
            for i in range(len(data) - window_size):
                X.append(data[i:i+window_size])
                y.append(data[i+window_size, self.close_idx])  # Use the stored close index
                
            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error creating sequences: {str(e)}")
    
    def inverse_transform_price(self, scaled_price: float, original_data: pd.DataFrame) -> float:
        """Convert scaled price back to the original scale"""
        try:
            # Create a dummy array with the same shape as the original data
            dummy = np.zeros((1, len(self.feature_columns)))
            dummy[0, self.close_idx] = scaled_price  # Use the stored close index
            
            # Inverse transform and extract the Close price
            return self.scaler.inverse_transform(dummy)[0, self.close_idx]
        except Exception as e:
            logger.error(f"Error inverse transforming price: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error inverse transforming price: {str(e)}")

class StockPredictor:
    def __init__(self):
        logger.info("Initializing StockPredictor")
        self.pipeline = DataPipeline()
        self.model = self._create_model()
        
    def _create_model(self) -> GRUModel:
        """Create a new model or load an existing one"""
        try:
            # Define model parameters
            input_dim = 8
            hidden_dim = 64 
            num_layers = 2
            output_dim = 1
            
            logger.info(f"Creating model with input_dim={input_dim}, hidden_dim={hidden_dim}")
            
            # Create a new model
            model = GRUModel(input_dim, hidden_dim, num_layers, output_dim)
            
            # Set model to evaluation mode
            model.eval()
            
            return model
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error creating model: {str(e)}")
    
    def predict(self, ticker: str, forecast_days: int = 5, window_size: int = 20) -> Dict[str, Any]:
        """Generate stock price predictions"""
        try:
            logger.info(f"Predicting {ticker} for {forecast_days} days with window_size={window_size}")
            
            # Fetch historical data
            stock_data = self.pipeline.fetch_stock_data(ticker)
            
            # Extract dates for the result
            dates = stock_data.index.strftime('%Y-%m-%d').tolist()
            
            # Preprocess data with technical indicators
            scaled_data, original_data = self.pipeline.preprocess_data(stock_data)
            
            # Generate prediction dates
            last_date = stock_data.index[-1]
            prediction_dates = [
                (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                for i in range(forecast_days)
            ]
            
            # Get number of features from the preprocessed data
            num_features = scaled_data.shape[1]
            logger.info(f"Number of features for prediction: {num_features}")
            
            # Prepare input for prediction
            X_pred = scaled_data[-window_size:].reshape(1, window_size, num_features)
            
            # Initialize predictions list
            predictions = []
            
            # Current input for step-by-step prediction
            current_input = X_pred.copy()
            
            # Make predictions for each day
            with torch.no_grad():
                for i in range(forecast_days):
                    # Convert input to tensor
                    input_tensor = torch.tensor(current_input, dtype=torch.float32)
                    
                    # Get prediction for the next day
                    pred = self.model(input_tensor).numpy()[0, 0]
                    predictions.append(pred)
                    logger.info(f"Day {i+1} prediction (scaled): {pred}")
                    
                    # Create a new row with the same values as the last row
                    new_pred_row = current_input[0, -1, :].copy()
                    
                    # Update the Close price
                    new_pred_row[self.pipeline.close_idx] = pred
                    
                    # Update current_input by removing the first row and adding the new prediction
                    current_input = np.roll(current_input, -1, axis=1)
                    current_input[0, -1, :] = new_pred_row
            
            # Convert scaled predictions back to original scale
            original_predictions = [
                self.pipeline.inverse_transform_price(pred, original_data)
                for pred in predictions
            ]
            logger.info(f"Original scale predictions: {original_predictions}")
            
            # Prepare historical data for the response
            historical_dict = {
                'dates': dates[-30:],  # Last 30 days
                'close': original_data['Close'].values[-30:].tolist(),
                'volume': original_data['Volume'].values[-30:].tolist()
            }
            
            # Return the result
            result = {
                'ticker': ticker,
                'historical_data': historical_dict,
                'predictions': original_predictions,
                'dates': dates[-30:],
                'prediction_dates': prediction_dates
            }
            logger.info("Prediction completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Prediction error: {str(e)}")

# Create an empty __init__.py file to make the directory a package
with open(os.path.join(os.path.dirname(__file__), "__init__.py"), "w") as f:
    pass 

# test the model
if __name__ == "__main__":
    predictor = StockPredictor()
    result = predictor.predict("AAPL", 5, 20)
    print(result)