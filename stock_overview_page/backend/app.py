from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any, Optional

app = FastAPI(title="Market Oracle API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockResponse(BaseModel):
    symbol: str
    current_price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    high_52_week: Optional[float] = None
    low_52_week: Optional[float] = None
    error: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to Market Oracle API"}

@app.get("/api/stock/{symbol}", response_model=StockResponse)
async def get_stock_data(symbol: str):
    try:
        # Fetch stock data
        ticker = yf.Ticker(symbol)
        ticker_data = ticker.history(period="1d")
        
        if ticker_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        
        # Get the latest price and other info
        info = ticker.info
        current_price = info.get('currentPrice', ticker_data['Close'].iloc[-1])
        previous_close = info.get('previousClose', ticker_data['Close'].iloc[-2] if len(ticker_data) > 1 else current_price)
        
        # Calculate change and percent change
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
        
        # Get additional data
        volume = info.get('volume', 0)
        market_cap = info.get('marketCap')
        high_52_week = info.get('fiftyTwoWeekHigh')
        low_52_week = info.get('fiftyTwoWeekLow')
        
        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "volume": volume,
            "market_cap": market_cap,
            "high_52_week": high_52_week,
            "low_52_week": low_52_week,
            "error": None
        }
    except Exception as e:
        # Return a partial response with error details
        return {
            "symbol": symbol.upper(),
            "current_price": 0.0,
            "change": 0.0,
            "change_percent": 0.0,
            "volume": 0,
            "error": str(e)
        }

@app.get("/api/stocks/search")
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 