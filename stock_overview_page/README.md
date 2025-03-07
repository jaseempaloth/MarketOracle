# Market Oracle

A full-stack web application for retrieving real-time stock market data.

## Features

- React frontend with responsive design
- FastAPI backend
- Real-time stock data retrieval using yfinance
- Stock symbol search functionality
- Data processing with pandas

## Project Structure

```
market_oracle/
├── backend/         # FastAPI server
│   ├── app.py       # Main application file
│   └── requirements.txt  # Python dependencies
└── frontend/        # React application
    ├── public/      # Static files
    ├── src/         # React source code
    └── package.json # Node dependencies
```

## Setup

### Backend

1. Navigate to the backend directory:

```
cd backend
```

2. Create and activate a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the FastAPI server:

```
uvicorn app:app --reload
```

### Frontend

1. Navigate to the frontend directory:

```
cd frontend
```

2. Install dependencies:

```
npm install
```

3. Start the development server:

```
npm start
```

4. Open your browser and go to `http://localhost:3000`

## API Endpoints

- `GET /api/stock/{symbol}`: Get real-time stock data for the specified symbol
