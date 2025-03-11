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
    ticker: str
    days: int = 5
    window_size: int = 20
    
class PredictionResponse(BaseModel):
    ticker: str
    predictions: List[float]
    prediction_dates: List[str]
    last_price: float

@router.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """
    Predict stock prices for the given ticker.
    
    - **ticker**: Stock ticker symbol (e.g., AAPL, MSFT)
    - **days**: Number of days to predict (default: 5)
    - **window_size**: Window size for the model input (default: 20)
    """
    try:
        if request.days < 1 or request.days > 30:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 30")
        
        if request.window_size < 5 or request.window_size > 100:
            raise HTTPException(status_code=400, detail="Window size must be between 5 and 100")
        
        # Get predictions
        result = stock_predictor.predict(request.ticker, request.days, request.window_size)
        
        # Simplify the response
        return PredictionResponse(
            ticker=request.ticker,
            predictions=result["predictions"],
            prediction_dates=result["prediction_dates"],
            last_price=result["historical_data"]["close"][-1] if "close" in result["historical_data"] else 0.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/stocks/search", response_model=Dict[str, List[Dict[str, str]]])
async def search_stocks(q: str = Query(..., description="Search query")):
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
            if q.lower() in stock["ticker"].lower() or q.lower() in stock["name"].lower()
        ]
        
        return {"results": filtered_stocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}") 