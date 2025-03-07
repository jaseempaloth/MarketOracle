import React from 'react';

const StockCard = ({ stockData }) => {
  if (!stockData) return null;

  const { 
    symbol, 
    current_price, 
    change, 
    change_percent, 
    volume, 
    market_cap, 
    high_52_week, 
    low_52_week 
  } = stockData;

  const isPositive = change >= 0;
  const changeColor = isPositive ? 'text-green-600' : 'text-red-600';
  const changeIcon = isPositive ? '▲' : '▼';

  // Format market cap to display in billions/millions
  const formatMarketCap = (cap) => {
    if (!cap) return 'N/A';
    if (cap >= 1e12) return `$${(cap / 1e12).toFixed(2)}T`;
    if (cap >= 1e9) return `$${(cap / 1e9).toFixed(2)}B`;
    if (cap >= 1e6) return `$${(cap / 1e6).toFixed(2)}M`;
    return `$${cap.toLocaleString()}`;
  };

  // Format volume with commas
  const formatVolume = (vol) => {
    return vol ? vol.toLocaleString() : 'N/A';
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 max-w-2xl w-full">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h2 className="text-2xl font-bold">{symbol}</h2>
        </div>
        <div className="text-right">
          <div className="text-3xl font-bold">${current_price.toFixed(2)}</div>
          <div className={`flex items-center justify-end ${changeColor}`}>
            <span className="mr-1">{changeIcon}</span>
            <span>${Math.abs(change).toFixed(2)} ({Math.abs(change_percent).toFixed(2)}%)</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="bg-gray-50 p-3 rounded">
          <div className="text-gray-500 text-sm">Volume</div>
          <div className="font-medium">{formatVolume(volume)}</div>
        </div>
        <div className="bg-gray-50 p-3 rounded">
          <div className="text-gray-500 text-sm">Market Cap</div>
          <div className="font-medium">{formatMarketCap(market_cap)}</div>
        </div>
        <div className="bg-gray-50 p-3 rounded">
          <div className="text-gray-500 text-sm">52-Week High</div>
          <div className="font-medium">{high_52_week ? `$${high_52_week.toFixed(2)}` : 'N/A'}</div>
        </div>
        <div className="bg-gray-50 p-3 rounded">
          <div className="text-gray-500 text-sm">52-Week Low</div>
          <div className="font-medium">{low_52_week ? `$${low_52_week.toFixed(2)}` : 'N/A'}</div>
        </div>
      </div>
    </div>
  );
};

export default StockCard; 