from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from ..models.stock_model import StockPredictor

router = APIRouter()

# Initialize stock predictor
stock_predictor = StockPredictor()

class PredictionRequest(BaseModel):
    days: int = 5
    window_size: int = 20
    
class PredictionResponse(BaseModel):
    ticker: str
    historical_data: Dict[str, List[float]]
    predictions: List[float]
    dates: List[str]
    prediction_dates: List[str]

@router.get("/predict/{ticker}", response_model=PredictionResponse)
async def predict_stock(
    ticker: str,
    days: int = Query(5, ge=1, le=30, description="Number of days to predict"),
    window_size: int = Query(20, ge=5, le=100, description="Window size for prediction")
):
    """
    Predict stock prices for the given ticker.
    
    - **ticker**: Stock ticker symbol (e.g., AAPL, MSFT)
    - **days**: Number of days to predict
    - **window_size**: Window size for the model input
    """
    try:
        # Get historical data and predictions
        result = stock_predictor.predict(ticker, days, window_size)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/stocks/search")
async def search_stocks(query: str = Query(..., min_length=1)):
    """
    Search for stocks based on the query string.
    """
    try:
        # This is a simplified version. In a real app, you would query a database
        # or an external API to get stock information.
        stocks = [
            {"ticker": "AAPL", "name": "Apple Inc."},
            {"ticker": "MSFT", "name": "Microsoft Corporation"},
            {"ticker": "GOOGL", "name": "Alphabet Inc."},
            {"ticker": "AMZN", "name": "Amazon.com, Inc."},
            {"ticker": "META", "name": "Meta Platforms, Inc."},
            {"ticker": "TSLA", "name": "Tesla, Inc."},
            {"ticker": "NVDA", "name": "NVIDIA Corporation"},
            {"ticker": "JPM", "name": "JPMorgan Chase & Co."},
            {"ticker": "V", "name": "Visa Inc."},
            {"ticker": "JNJ", "name": "Johnson & Johnson"}
        ]
        
        filtered_stocks = [
            stock for stock in stocks 
            if query.lower() in stock["ticker"].lower() or query.lower() in stock["name"].lower()
        ]
        
        return {"results": filtered_stocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}") 