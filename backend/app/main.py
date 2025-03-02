from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .routers import predictions

app = FastAPI(
    title="Stock Prediction API",
    description="API for predicting stock prices using a GRU-based RNN model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api", tags=["predictions"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Stock Prediction API",
        "docs": "/docs",
        "endpoints": {
            "predictions": "/api/predict/{ticker}"
        }
    }

# Add this section to allow running the file directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 