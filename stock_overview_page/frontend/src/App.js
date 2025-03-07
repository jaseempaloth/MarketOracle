import React, { useState } from 'react';
import SearchBar from './components/SearchBar';
import StockCard from './components/StockCard';
import ErrorMessage from './components/ErrorMessage';
import { fetchStockData } from './services/api';

function App() {
  const [stockData, setStockData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async (symbol) => {
    setLoading(true);
    setError('');
    
    try {
      const data = await fetchStockData(symbol);
      
      if (data.error) {
        setError(data.error);
        setStockData(null);
      } else {
        setStockData(data);
      }
    } catch (err) {
      setError(err.message || 'An error occurred while fetching stock data');
      setStockData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center">
      <header className="w-full bg-primary text-white shadow-md py-6">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold text-center">Market Oracle</h1>
          <p className="text-center mt-2 text-gray-300">Get real-time stock market data</p>
        </div>
      </header>
      
      <main className="container mx-auto px-4 py-8 flex flex-col items-center justify-center gap-6 flex-1">
        <div className="w-full max-w-2xl flex flex-col items-center gap-6">
          <SearchBar onSearch={handleSearch} isLoading={loading} />
          
          {loading && !error && (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
              <p className="mt-4 text-gray-600">Loading stock data...</p>
            </div>
          )}
          
          <ErrorMessage message={error} />
          
          {stockData && !loading && !error && (
            <StockCard stockData={stockData} />
          )}
          
          {!stockData && !loading && !error && (
            <div className="bg-white rounded-lg shadow-md p-8 text-center max-w-2xl w-full">
              <svg className="w-16 h-16 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <p className="mt-4 text-gray-600">Enter a stock symbol to view real-time market data</p>
              <div className="mt-4">
                <p className="text-sm text-gray-500">Example symbols: AAPL, MSFT, GOOGL, AMZN, TSLA</p>
              </div>
            </div>
          )}
        </div>
      </main>
      
      <footer className="w-full bg-gray-100 py-4 border-t border-gray-200 mt-auto">
        <div className="container mx-auto px-4">
          <p className="text-center text-gray-600 text-sm">
            &copy; {new Date().getFullYear()} MarketOracle.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App; 