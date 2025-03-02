import axios from 'axios';

// Base URL for API requests - update to include the full URL in development
const API_BASE_URL = 'http://localhost:8000/api';

/**
 * Search for stocks based on a query string
 * @param {string} query - The search query
 * @returns {Promise} - Promise resolving to the search results
 */
export const searchStocks = async (query) => {
  try {
    console.log(`Searching for stocks with query: ${query}`);
    const response = await axios.get(`${API_BASE_URL}/stocks/search`, {
      params: { query }
    });
    return response.data.results;
  } catch (error) {
    console.error('Error searching stocks:', error);
    throw new Error(error.response?.data?.detail || 'Failed to search stocks');
  }
};

/**
 * Fetch stock price predictions
 * @param {string} ticker - The stock ticker symbol
 * @param {number} days - Number of days to predict
 * @param {number} windowSize - Window size for prediction
 * @returns {Promise} - Promise resolving to the prediction data
 */
export const fetchPrediction = async (ticker, days = 5, windowSize = 20) => {
  try {
    console.log(`Fetching prediction for ${ticker} with days=${days}, windowSize=${windowSize}`);
    const response = await axios.get(`${API_BASE_URL}/predict/${ticker}`, {
      params: { days, window_size: windowSize }
    });
    console.log('Prediction response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching prediction:', error);
    const errorMessage = error.response?.data?.detail || 'Failed to fetch prediction';
    console.error('Error details:', errorMessage);
    throw new Error(errorMessage);
  }
}; 