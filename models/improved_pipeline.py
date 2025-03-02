import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import logging
import traceback
import joblib
import requests
import os
import math

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("improved_pipeline")

class ImprovedDataPipeline:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        
    def fetch_stock_data(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """Fetch historical stock data with longer period by default"""
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
    
    def fetch_market_data(self, period: str = "5y") -> pd.DataFrame:
        """Fetch market index data for additional context"""
        try:
            logger.info("Fetching market index data")
            # Get S&P 500 data
            sp500 = yf.download("^GSPC", period=period)
            # Get VIX volatility index
            vix = yf.download("^VIX", period=period)
            
            # Make sure we're working with the Close prices
            sp500_close = sp500['Close'] if 'Close' in sp500.columns else sp500['Adj Close']
            vix_close = vix['Close'] if 'Close' in vix.columns else vix['Adj Close']
            
            # Create a DataFrame with proper indexing
            market_data = pd.DataFrame(index=sp500.index)
            market_data['SP500'] = sp500_close
            market_data['VIX'] = vix_close.reindex(market_data.index, method='ffill')
            
            logger.info(f"Successfully fetched market data with {len(market_data)} rows")
            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            traceback.print_exc()
            # Return empty DataFrame but don't fail - we can still proceed without market data
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced technical indicators for feature engineering"""
        try:
            logger.info("Calculating technical indicators")
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Inspect structure of columns
            logger.info(f"DataFrame column types: {type(df.columns)}")
            logger.info(f"DataFrame columns: {df.columns}")
            
            # Handle potential multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                logger.info("Detected multi-level columns, flattening")
                # Get the levels and names
                logger.info(f"MultiIndex levels: {df.columns.names}")
                
                # First try: check if 'Adj Close' is in the first level
                try:
                    # For a typical YF MultiIndex like (column_name, ticker_name)
                    col_names = df.columns.get_level_values(0).unique()
                    logger.info(f"Column level 0 values: {col_names}")
                    
                    # Create a new DataFrame with flattened column names
                    new_df = pd.DataFrame(index=df.index)
                    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                        if col in col_names:
                            # Extract this column for the first ticker
                            new_df[col] = df[col].iloc[:, 0]
                    
                    # Check if we got what we needed
                    if 'Close' not in new_df.columns and 'Adj Close' in new_df.columns:
                        new_df['Close'] = new_df['Adj Close']
                    
                    # Replace df with our new flattened DataFrame
                    df = new_df
                    logger.info(f"Successfully flattened columns: {df.columns}")
                except Exception as e1:
                    logger.error(f"Error in first flattening approach: {str(e1)}")
                    # Second approach: try direct droplevel
                    try:
                        df.columns = df.columns.droplevel(1)  # Drop the ticker level
                        logger.info(f"Flattened columns using droplevel: {df.columns}")
                    except Exception as e2:
                        logger.error(f"Error in second flattening approach: {str(e2)}")
                        # Third approach: directly convert to strings and recreate
                        try:
                            col_strings = [f"{c[0]}_{c[1]}" for c in df.columns]
                            df.columns = col_strings
                            logger.info(f"Flattened columns to strings: {df.columns}")
                            
                            # Try to identify key columns by partial matching
                            close_cols = [c for c in df.columns if 'Close' in c]
                            open_cols = [c for c in df.columns if 'Open' in c]
                            high_cols = [c for c in df.columns if 'High' in c]
                            low_cols = [c for c in df.columns if 'Low' in c]
                            vol_cols = [c for c in df.columns if 'Volume' in c]
                            
                            # Create standard columns from what we found
                            if close_cols:
                                df['Close'] = df[close_cols[0]]
                            if open_cols:
                                df['Open'] = df[open_cols[0]]
                            if high_cols:
                                df['High'] = df[high_cols[0]]
                            if low_cols:
                                df['Low'] = df[low_cols[0]]
                            if vol_cols:
                                df['Volume'] = df[vol_cols[0]]
                                
                        except Exception as e3:
                            logger.error(f"Error in third flattening approach: {str(e3)}")
                            # Last resort: extract by position
                            logger.info("Using last resort: extracting by position")
                            try:
                                new_df = pd.DataFrame(index=df.index)
                                new_df['Open'] = df.iloc[:, 0]
                                new_df['High'] = df.iloc[:, 1]
                                new_df['Low'] = df.iloc[:, 2]
                                new_df['Close'] = df.iloc[:, 3]
                                if df.shape[1] > 4:
                                    new_df['Volume'] = df.iloc[:, 5]
                                df = new_df
                            except Exception as e4:
                                logger.error(f"Error in last resort approach: {str(e4)}")
                                # Give up and raise an error
                                raise Exception("Could not parse yfinance DataFrame structure")
            
            # Ensure we have all required columns
            required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Required column {col} not found, using Close as substitute")
                    df[col] = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
            
            # Ensure all columns are Series (not DataFrames)
            for col in df.columns:
                if hasattr(df[col], 'values') and isinstance(df[col].values, np.ndarray):
                    if len(df[col].values.shape) > 1 and df[col].values.shape[1] > 1:
                        logger.info(f"Converting multi-column series {col} to single column")
                        df[col] = df[col].iloc[:, 0]
            
            logger.info(f"Final columns before calculation: {df.columns}")
            
            # Ensure all data is numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values from numeric conversion
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Simple Moving Averages (SMA)
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Fill NaN values from the indicators
            df = df.fillna(method='bfill').fillna(0)
            
            # Exponential Moving Averages (EMA)
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
            
            # Price to MA ratios - indicators of momentum
            # Ensure we're working with Series, not DataFrames
            close_series = df['Close'].squeeze() if hasattr(df['Close'], 'squeeze') else df['Close']
            sma_200_series = df['SMA_200'].squeeze() if hasattr(df['SMA_200'], 'squeeze') else df['SMA_200']
            sma_50_series = df['SMA_50'].squeeze() if hasattr(df['SMA_50'], 'squeeze') else df['SMA_50']
            
            # Safely calculate price to MA ratios
            df['Price_to_SMA_200'] = close_series / sma_200_series.replace(0, np.nan).fillna(1)
            df['Price_to_SMA_50'] = close_series / sma_50_series.replace(0, np.nan).fillna(1)
            
            # MA Crossovers
            df['SMA_5_10_Crossover'] = df['SMA_5'] - df['SMA_10']
            df['SMA_10_20_Crossover'] = df['SMA_10'] - df['SMA_20']
            df['SMA_50_200_Crossover'] = df['SMA_50'] - df['SMA_200']  # Golden/Death Cross
            
            # Handle potential NaN values
            df = df.fillna(method='bfill').fillna(0)
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
            
            # Handle division safely
            bb_diff = df['BB_Upper'] - df['BB_Lower']
            df['BB_Width'] = (bb_diff / df['BB_Middle'].replace(0, np.nan)).fillna(0)
            
            # Handle division safely for BB_Position - avoid division by zero
            bb_range = df['BB_Upper'] - df['BB_Lower']
            # Where range is zero, use 0.5 (middle of the band)
            df['BB_Position'] = np.where(
                bb_range > 0,
                (df['Close'] - df['BB_Lower']) / bb_range,
                0.5
            )
            
            # Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Apply rolling mean with min_periods to handle initial NaNs
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            
            # Handle division by zero more carefully - use filled_ewma
            # Where avg_loss is 0, RS is set to 100 (equivalent to RSI=100)
            rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
            
            # Calculate RSI
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Clip RSI to valid range 0-100
            df['RSI'] = np.clip(df['RSI'], 0, 100)
            
            # RSI divergence (difference between price direction and RSI direction)
            df['RSI_Diff'] = df['RSI'].diff(5)
            df['Close_Diff'] = df['Close'].diff(5)
            
            # Calculate RSI divergence safely
            rsi_sign = np.sign(df['RSI_Diff'])
            close_sign = np.sign(df['Close_Diff'])
            
            # Convert boolean to int, handling NaNs
            df['RSI_Divergence'] = np.where(
                ~np.isnan(rsi_sign) & ~np.isnan(close_sign),
                (rsi_sign != close_sign).astype(int),
                0
            )
            
            # MACD (Moving Average Convergence Divergence)
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            df['MACD_Crossover'] = df['MACD'] - df['MACD_Signal']
            
            # Stochastic Oscillator - handle min/max more carefully
            low_14 = df['Low'].rolling(window=14, min_periods=1).min()
            high_14 = df['High'].rolling(window=14, min_periods=1).max()
            
            # Calculate K% safely
            high_low_diff = high_14 - low_14
            # When high and low are the same, K% is 50
            df['K_Percent'] = np.where(
                high_low_diff > 0,
                100 * ((df['Close'] - low_14) / high_low_diff),
                50
            )
            
            df['D_Percent'] = df['K_Percent'].rolling(window=3, min_periods=1).mean()
            
            # Clip to valid range 0-100
            df['K_Percent'] = np.clip(df['K_Percent'], 0, 100)
            df['D_Percent'] = np.clip(df['D_Percent'], 0, 100)
            
            # Calculate ADX (Average Directional Index) safely
            try:
                # Calculate True Range first
                tr1 = df['High'] - df['Low']
                tr2 = (df['High'] - df['Close'].shift(1)).abs()
                tr3 = (df['Low'] - df['Close'].shift(1)).abs()
                
                # Combine to get True Range
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # Calculate Average True Range
                atr = tr.rolling(window=14, min_periods=1).mean()
                
                # Direction Movement
                plus_dm = df['High'].diff().clip(lower=0)
                minus_dm = df['Low'].diff().multiply(-1).clip(lower=0)
                
                # Directional Indicators - handle division by zero
                plus_di = np.where(
                    atr > 0, 
                    100 * (plus_dm.rolling(window=14, min_periods=1).mean() / atr),
                    0
                )
                
                minus_di = np.where(
                    atr > 0,
                    100 * (minus_dm.rolling(window=14, min_periods=1).mean() / atr),
                    0
                )
                
                # Calculate DX safely
                di_sum = np.abs(plus_di) + np.abs(minus_di)
                dx = np.where(
                    di_sum > 0,
                    100 * (np.abs(plus_di - minus_di) / di_sum),
                    0
                )
                
                # Calculate ADX
                df['ADX'] = pd.Series(dx).rolling(window=14, min_periods=1).mean()
                df['ADX'] = df['ADX'].fillna(0).clip(0, 100)  # Clip to valid range
            except Exception as adx_error:
                logger.error(f"Error calculating ADX: {str(adx_error)}")
                df['ADX'] = 50  # Use neutral value
            
            # Safe Volume indicators
            df['Volume_SMA_5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            
            # Volume Ratio - handle division safely
            df['Volume_Ratio'] = np.where(
                df['Volume_SMA_20'] > 0,
                df['Volume'] / df['Volume_SMA_20'],
                1
            )
            
            # On-Balance Volume (OBV) - handle NaNs in diff
            df['OBV'] = np.where(
                df['Close'].diff() > 0,
                df['Volume'],
                np.where(
                    df['Close'].diff() < 0,
                    -df['Volume'],
                    0
                )
            ).cumsum()
            
            # Add day of week and month features for seasonality
            try:
                df['DayOfWeek'] = df.index.dayofweek
                df['Month'] = df.index.month
                df['Quarter'] = df.index.quarter
            except Exception as e:
                logger.error(f"Error creating calendar features: {str(e)}")
                # Create default values if index doesn't support these
                df['DayOfWeek'] = 0
                df['Month'] = 1
                df['Quarter'] = 1
            
            # Daily Returns and historical volatility - handle division safely
            df['Daily_Return'] = df['Close'].pct_change().fillna(0)
            
            # Add lagged returns
            df['Daily_Return_Lag1'] = df['Daily_Return'].shift(1).fillna(0)
            df['Daily_Return_Lag2'] = df['Daily_Return'].shift(2).fillna(0)
            df['Daily_Return_Lag3'] = df['Daily_Return'].shift(3).fillna(0)
            df['Daily_Return_Lag5'] = df['Daily_Return'].shift(5).fillna(0)
            
            # Volatility (standard deviation of returns over different windows)
            df['Volatility_10d'] = df['Daily_Return'].rolling(window=10, min_periods=1).std().fillna(0)
            df['Volatility_21d'] = df['Daily_Return'].rolling(window=21, min_periods=1).std().fillna(0)
            df['Volatility_63d'] = df['Daily_Return'].rolling(window=63, min_periods=1).std().fillna(0)
            
            # Rate of change - safer calculation
            df['ROC_5'] = 100 * (df['Close'] / df['Close'].shift(5).replace(0, np.nan) - 1).fillna(0)
            df['ROC_10'] = 100 * (df['Close'] / df['Close'].shift(10).replace(0, np.nan) - 1).fillna(0)
            df['ROC_21'] = 100 * (df['Close'] / df['Close'].shift(21).replace(0, np.nan) - 1).fillna(0)
            
            # Price channels
            df['Highest_High_20d'] = df['High'].rolling(window=20, min_periods=1).max()
            df['Lowest_Low_20d'] = df['Low'].rolling(window=20, min_periods=1).min()
            
            # Channel Width - safer calculation
            df['Channel_Width'] = np.where(
                df['Close'] > 0,
                (df['Highest_High_20d'] - df['Lowest_Low_20d']) / df['Close'],
                0
            )
            
            # Fill any remaining NaN values
            df = df.fillna(0)
            
            logger.info("Technical indicators calculated successfully")
            return df
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            traceback.print_exc()
            
            # Try a minimal set of indicators if full calculation fails
            try:
                logger.info("Attempting minimal indicator calculation")
                simple_df = data.copy()
                
                # Handle MultiIndex columns
                if isinstance(simple_df.columns, pd.MultiIndex):
                    # Extract first column as Close
                    simple_df = pd.DataFrame(index=data.index)
                    simple_df['Close'] = data.iloc[:, 0]
                    simple_df['Open'] = data.iloc[:, 0]  # Fallback
                    simple_df['High'] = data.iloc[:, 0]  # Fallback
                    simple_df['Low'] = data.iloc[:, 0]   # Fallback
                    simple_df['Volume'] = 0              # Fallback
                
                # Ensure required columns exist
                for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
                    if col not in simple_df.columns:
                        simple_df[col] = simple_df.iloc[:, 0] if len(simple_df.columns) > 0 else 0
                
                # Calculate just a few basic indicators
                simple_df['SMA_5'] = simple_df['Close'].rolling(window=5, min_periods=1).mean().fillna(0)
                simple_df['SMA_20'] = simple_df['Close'].rolling(window=20, min_periods=1).mean().fillna(0)
                simple_df['Daily_Return'] = simple_df['Close'].pct_change().fillna(0)
                
                # These are less likely to cause errors
                try:
                    simple_df['RSI'] = 50  # Neutral value
                    simple_df['MACD'] = 0  # Neutral value
                    simple_df['MACD_Hist'] = 0  # Neutral value
                    simple_df['BB_Width'] = 0 
                    simple_df['BB_Position'] = 0.5  # Middle of band
                except Exception:
                    pass
                
                # Return the simplified dataframe
                logger.info("Minimal indicators calculated as fallback")
                return simple_df.fillna(0)
                
            except Exception as minimal_error:
                logger.error(f"Error in minimal indicator calculation: {str(minimal_error)}")
                # Ultimate fallback - just return the input data with required columns
                final_fallback = pd.DataFrame(index=data.index)
                
                # Try to extract close price
                try:
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        final_fallback['Close'] = data.iloc[:, 0]
                    else:
                        final_fallback['Close'] = 0
                except:
                    final_fallback['Close'] = 0
                
                # Add other required columns
                final_fallback['Open'] = final_fallback['Close']
                final_fallback['High'] = final_fallback['Close']
                final_fallback['Low'] = final_fallback['Close']
                final_fallback['Volume'] = 0
                final_fallback['SMA_5'] = 0
                final_fallback['SMA_20'] = 0
                final_fallback['RSI'] = 50
                
                logger.warning("Using emergency fallback data with minimal columns")
                return final_fallback
    
    def preprocess_data(self, data: pd.DataFrame, market_data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """Preprocess the stock data with technical indicators and optional market data"""
        try:
            logger.info("Preprocessing data")
            # Calculate technical indicators
            data_with_indicators = self.calculate_technical_indicators(data)
            
            # Combine with market data if available
            if market_data is not None and not market_data.empty:
                try:
                    logger.info("Adding market data features")
                    # Align indices
                    aligned_market_data = market_data.reindex(data.index)
                    # For any NaN values in aligned data, use forward fill, then backward fill
                    aligned_market_data = aligned_market_data.ffill().bfill()
                    
                    # Calculate additional market ratios
                    if 'SP500' in aligned_market_data.columns:
                        # Ensure we're working with Series for calculations
                        sp500_series = aligned_market_data['SP500'].squeeze()
                        close_series = data_with_indicators['Close'].squeeze() 
                        
                        # Calculate price to SP500 ratio where SP500 is not zero
                        data_with_indicators['Price_to_SP500'] = (close_series / 
                            sp500_series.where(sp500_series > 0, np.nan)).fillna(1)
                        
                        # Calculate SP500 returns for correlation
                        aligned_market_data['SP500_Return'] = sp500_series.pct_change()
                        
                        # Calculate 63-day rolling correlation with market returns
                        if len(data_with_indicators) > 63:  # Only if we have enough data
                            stock_returns = data_with_indicators['Daily_Return']
                            market_returns = aligned_market_data['SP500_Return']
                            
                            # Calculate correlation, handling NaN values
                            rolling_corr = stock_returns.rolling(63).corr(market_returns)
                            data_with_indicators['Market_Correlation_63d'] = rolling_corr
                    
                    # Add VIX as volatility indicator if available
                    if 'VIX' in aligned_market_data.columns:
                        data_with_indicators['VIX'] = aligned_market_data['VIX']
                        
                except Exception as market_error:
                    logger.error(f"Error integrating market data, continuing without it: {str(market_error)}")
                    logger.info("Continuing with stock data only")
            
            # Handle missing values created by indicators that use lookback periods
            data_with_indicators = data_with_indicators.bfill().ffill()
            
            # Define core features that we'll always include
            core_features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_5', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'RSI', 'MACD', 'MACD_Hist', 'BB_Width', 'BB_Position',
                'K_Percent', 'D_Percent', 'ADX', 'Volatility_21d',
                'Daily_Return', 'ROC_5', 'ROC_21', 'Volume_Ratio'
            ]
            
            # Add market features if they were successfully created
            market_features = ['Price_to_SP500', 'VIX', 'Market_Correlation_63d']
            for feat in market_features:
                if feat in data_with_indicators.columns:
                    core_features.append(feat)
            
            # Ensure all selected features exist
            available_features = [col for col in core_features if col in data_with_indicators.columns]
            logger.info(f"Using features: {available_features}")
            
            # Store feature columns for later use
            self.feature_columns = available_features
            
            # Get the processed data with selected features
            processed_data = data_with_indicators[available_features].copy()
            
            # Store the column indices for reference
            self.column_indices = {col: i for i, col in enumerate(processed_data.columns)}
            self.close_idx = self.column_indices.get('Close', 3)  # Default to 3 if not found
            logger.info(f"Close price index: {self.close_idx}")
            
            # Replace any remaining NaN values with 0
            processed_data = processed_data.fillna(0)
            
            # Feature scaling
            scaled_data = self.scaler.fit_transform(processed_data)
            logger.info("Data preprocessing completed successfully")
            
            return scaled_data, processed_data
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error preprocessing data: {str(e)}")
    
    def create_sequences(self, data: np.ndarray, window_size: int, step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences for the model with optional overlapping windows"""
        try:
            X, y = [], []
            for i in range(0, len(data) - window_size, step_size):
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
            
    def split_data(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple:
        """Split data into training and validation sets with a walk-forward approach"""
        try:
            split_idx = int(len(X) * train_ratio)
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
            return X_train, X_val, y_train, y_val, split_idx
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error splitting data: {str(e)}")
            
    def prepare_tensors(self, X_train, X_val, y_train, y_val, device):
        """Convert NumPy arrays to PyTorch tensors"""
        try:
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)
            
            logger.info("Data converted to tensors successfully")
            return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor
        except Exception as e:
            logger.error(f"Error preparing tensors: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error preparing tensors: {str(e)}")

# For compatibility with torch imports in notebook
try:
    import torch
except ImportError:
    logger.warning("PyTorch not available for tensor preparation") 