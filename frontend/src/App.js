import React, { useState } from 'react';
import './App.css';
import StockSearch from './components/StockSearch';
import StockChart from './components/StockChart';
import PredictionForm from './components/PredictionForm';
import { fetchPrediction } from './services/api';

function App() {
  const [ticker, setTicker] = useState('');
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = (selectedTicker) => {
    setTicker(selectedTicker);
    setPredictionData(null);
    setError(null);
  };

  const handlePrediction = async (days, windowSize) => {
    if (!ticker) {
      setError('Please select a stock ticker first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await fetchPrediction(ticker, days, windowSize);
      setPredictionData(data);
    } catch (err) {
      setError(`Error fetching prediction: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Stock Price Prediction</h1>
        <p>Powered by GRU-based RNN</p>
      </header>

      <main className="App-main">
        <section className="search-section">
          <StockSearch onSearch={handleSearch} />
        </section>

        {ticker && (
          <section className="prediction-section">
            <h2>Prediction for {ticker}</h2>
            <PredictionForm onSubmit={handlePrediction} isLoading={loading} />
            
            {error && <div className="error-message">{error}</div>}
            
            {loading && <div className="loading">Loading predictions...</div>}
            
            {predictionData && (
              <div className="prediction-results">
                <StockChart 
                  historicalData={predictionData.historical_data}
                  predictions={predictionData.predictions}
                  dates={predictionData.dates}
                  predictionDates={predictionData.prediction_dates}
                />
                
                <div className="prediction-values">
                  <h3>Predicted Values</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Predicted Close Price</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictionData.prediction_dates.map((date, index) => (
                        <tr key={date}>
                          <td>{date}</td>
                          <td>${predictionData.predictions[index].toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="App-footer">
        <p>© 2023 Stock Prediction App</p>
      </footer>
    </div>
  );
}

export default App; 