import React, { useState } from 'react';

const PredictionForm = ({ onSubmit, isLoading }) => {
  const [days, setDays] = useState(5);
  const [windowSize, setWindowSize] = useState(20);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(days, windowSize);
  };

  return (
    <div className="prediction-form">
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="days">
            Prediction Days:
            <span className="value-display">{days}</span>
          </label>
          <input
            type="range"
            id="days"
            min="1"
            max="30"
            value={days}
            onChange={(e) => setDays(parseInt(e.target.value))}
            disabled={isLoading}
          />
          <div className="range-labels">
            <span>1</span>
            <span>15</span>
            <span>30</span>
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="windowSize">
            Window Size:
            <span className="value-display">{windowSize}</span>
          </label>
          <input
            type="range"
            id="windowSize"
            min="5"
            max="100"
            step="5"
            value={windowSize}
            onChange={(e) => setWindowSize(parseInt(e.target.value))}
            disabled={isLoading}
          />
          <div className="range-labels">
            <span>5</span>
            <span>50</span>
            <span>100</span>
          </div>
        </div>

        <button 
          type="submit" 
          className="predict-button" 
          disabled={isLoading}
        >
          {isLoading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      <style jsx>{`
        .prediction-form {
          margin: 20px 0;
          padding: 15px;
          background-color: white;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .form-group {
          margin-bottom: 20px;
        }
        
        label {
          display: block;
          margin-bottom: 8px;
          font-weight: 500;
        }
        
        .value-display {
          margin-left: 10px;
          font-weight: bold;
          color: #007bff;
        }
        
        input[type="range"] {
          width: 100%;
          max-width: 400px;
        }
        
        .range-labels {
          display: flex;
          justify-content: space-between;
          max-width: 400px;
          margin-top: 5px;
          color: #666;
          font-size: 0.8rem;
        }
        
        .predict-button {
          background-color: #007bff;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 4px;
          font-size: 1rem;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .predict-button:hover:not(:disabled) {
          background-color: #0069d9;
        }
        
        .predict-button:disabled {
          background-color: #cccccc;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default PredictionForm; 