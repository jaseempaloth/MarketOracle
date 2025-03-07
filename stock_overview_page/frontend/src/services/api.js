import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const fetchStockData = async (symbol) => {
  try {
    const response = await axios.get(`${API_URL}/api/stock/${symbol}`);
    return response.data;
  } catch (error) {
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      throw new Error(error.response.data.detail || 'Error fetching stock data');
    } else if (error.request) {
      // The request was made but no response was received
      throw new Error('Server is not responding. Please try again later.');
    } else {
      // Something happened in setting up the request that triggered an Error
      throw new Error('Error setting up the request: ' + error.message);
    }
  }
}; 