# Stock Prediction Application - Market Oracle

A full-stack application that leverages a GRU-based RNN model to predict stock prices. The system includes a data processing pipeline, model training, FastAPI backend, and React frontend.

## Project Structure

- **models/**: Jupyter notebooks for model development and pipeline creation
- **backend/**: FastAPI backend service
- **frontend/**: React frontend application
- **data/**: Directory for storing historical stock data and trained models
- **utils/**: Utility functions and helpers for data processing and model evaluation
- **tests/**: Unit and integration tests
- **docs/**: Additional documentation

## Components

### 1. Model Development & Pipeline Creation

- User input & data acquisition using yfinance
- Data preprocessing pipeline (normalization, feature engineering, sequence creation)
- GRU-based RNN model built with PyTorch
- Hyperparameter tuning framework
- Model training and evaluation with cross-validation
- Performance metrics tracking (RMSE, MAE, MAPE)

### 2. Backend API with FastAPI

- RESTful API endpoints for prediction and historical data retrieval
- Asynchronous request handling
- Integration with the trained model
- Consistent data preprocessing
- Caching mechanism for frequently requested stocks
- Rate limiting and error handling
- Swagger/OpenAPI documentation

### 3. Frontend with React

- Interactive UI for user input
- Visualization of predictions and historical data using D3/Chart.js
- Responsive design for mobile and desktop
- State management with React Context/Redux
- Error handling and loading states
- User settings persistence

## Setup and Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm 6+
- Git

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/jaseempaloth/MarketOracle
cd MarketOracle

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Start the development server
uvicorn app.main:app --reload
```

The backend will be available at `http://localhost:8000`.

### Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at `http://localhost:3000`.

## Usage

1. Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
2. Select the desired time range for historical data
3. View historical price chart and predicted future prices
4. Customize prediction parameters:
   - Prediction horizon (1-30 days)
   - Confidence interval
   - Technical indicators to include

## Pipeline Architecture

The application uses a consistent pipeline across both training and inference to ensure reliable predictions:

1. Data acquisition: Fetching historical data from reliable financial APIs
2. Preprocessing: Normalization, feature engineering, sequence creation
3. Model inference: Forward pass through the trained GRU-RNN model
4. Post-processing: Denormalization, confidence interval calculation
5. Result formatting: JSON response or visualization-ready format

## Model Performance

The GRU-RNN model has been evaluated on various stocks and market conditions:

- Average RMSE: 2.3% on 5-day predictions
- Directional accuracy: 68% for next-day movement
- Performance varies by market volatility and stock-specific factors

## Deployment

### Docker Deployment

```bash
# Build and start the containers
docker-compose up -d

# Monitor logs
docker-compose logs -f
```

### Cloud Deployment

The application can be deployed to cloud platforms:

- **AWS**: Use Elastic Beanstalk or ECS for containerized deployment
- **Google Cloud**: App Engine or GKE
- **Azure**: Azure App Service or AKS

Detailed cloud deployment instructions can be found in the `docs/deployment.md` file.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for data acquisition
- [PyTorch](https://pytorch.org/) for deep learning capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for backend API
- [React](https://reactjs.org/) for frontend development
