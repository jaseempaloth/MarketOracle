import React, { useState } from 'react';
import Select from 'react-select';
import { searchStocks } from '../services/api';

const StockSearch = ({ onSearch }) => {
  const [inputValue, setInputValue] = useState('');
  const [options, setOptions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  
  // Handle input change
  const handleInputChange = async (value) => {
    setInputValue(value);
    
    if (value.length < 2) {
      setOptions([]);
      return;
    }
    
    setIsLoading(true);
    
    try {
      const results = await searchStocks(value);
      const selectOptions = results.map(stock => ({
        value: stock.ticker,
        label: `${stock.ticker} - ${stock.name}`
      }));
      setOptions(selectOptions);
    } catch (error) {
      console.error('Error searching stocks:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle selection
  const handleChange = (selectedOption) => {
    if (selectedOption) {
      onSearch(selectedOption.value);
    }
  };
  
  // Component styles
  const customStyles = {
    control: (provided) => ({
      ...provided,
      width: '100%',
      maxWidth: '500px',
      margin: '0 auto',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
      borderColor: '#ddd',
    }),
    menu: (provided) => ({
      ...provided,
      maxWidth: '500px',
      margin: '0 auto',
      zIndex: 10,
    }),
  };
  
  return (
    <div className="stock-search">
      <h2>Search for a Stock</h2>
      <p>Enter a stock ticker or company name</p>
      
      <Select
        inputValue={inputValue}
        onInputChange={handleInputChange}
        onChange={handleChange}
        options={options}
        isLoading={isLoading}
        placeholder="Search for a stock (e.g., AAPL, Microsoft)"
        isClearable
        isSearchable
        styles={customStyles}
        noOptionsMessage={() => 
          inputValue.length < 2 
            ? "Enter at least 2 characters to search" 
            : "No stocks found"
        }
      />
      
      <div className="stock-examples">
        <p>Examples: AAPL, MSFT, GOOGL, AMZN, TSLA</p>
      </div>
    </div>
  );
};

export default StockSearch; 