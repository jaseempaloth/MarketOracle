import React, { useState, useEffect } from 'react';

const SearchBar = ({ onSearch, isLoading }) => {
  const [symbol, setSymbol] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  useEffect(() => {
    const fetchSuggestions = async () => {
      if (symbol.trim().length > 0) {
        try {
          const response = await fetch(`http://localhost:8000/api/stocks/search?query=${encodeURIComponent(symbol.trim())}`);
          const data = await response.json();
          setSuggestions(data.results);
          setShowSuggestions(true);
        } catch (error) {
          console.error('Error fetching suggestions:', error);
          setSuggestions([]);
        }
      } else {
        setSuggestions([]);
        setShowSuggestions(false);
      }
    };

    const debounceTimer = setTimeout(fetchSuggestions, 300);
    return () => clearTimeout(debounceTimer);
  }, [symbol]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (symbol.trim()) {
      onSearch(symbol.trim().toUpperCase());
      setShowSuggestions(false);
    }
  };

  const handleSuggestionClick = (ticker) => {
    setSymbol(ticker);
    onSearch(ticker);
    setShowSuggestions(false);
  };

  return (
    <div className="w-full max-w-md">
      <form onSubmit={handleSubmit} className="flex items-center">
        <div className="relative w-full">
          <input
            type="text"
            className="block w-full p-3 pr-10 text-sm border border-gray-300 rounded-lg bg-white focus:ring-blue-500 focus:border-blue-500"
            placeholder="Enter stock symbol (e.g., AAPL)"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            disabled={isLoading}
          />
          {isLoading && (
            <div className="absolute right-3 top-3">
              <svg 
                className="animate-spin h-5 w-5 text-blue-500" 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24"
              >
                <circle 
                  className="opacity-25" 
                  cx="12" 
                  cy="12" 
                  r="10" 
                  stroke="currentColor" 
                  strokeWidth="4"
                ></circle>
                <path 
                  className="opacity-75" 
                  fill="currentColor" 
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
            </div>
          )}
          {showSuggestions && suggestions.length > 0 && (
            <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg">
              {suggestions.map((stock) => (
                <div
                  key={stock.ticker}
                  className="p-2 hover:bg-gray-100 cursor-pointer"
                  onClick={() => handleSuggestionClick(stock.ticker)}
                >
                  <span className="font-medium">{stock.ticker}</span> - {stock.name}
                </div>
              ))}
            </div>
          )}
        </div>
        <button
          type="submit"
          className="p-3 ml-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 focus:ring-4 focus:outline-none focus:ring-blue-300 disabled:opacity-50"
          disabled={isLoading || !symbol.trim()}
        >
          Search
        </button>
      </form>
    </div>
  );
};

export default SearchBar; 