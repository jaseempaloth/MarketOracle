import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

const StockChart = ({ historicalData, predictions, dates, predictionDates }) => {
  // Combining historical and predicted data for display
  const combinedDates = [...dates, ...predictionDates];
  
  // Get the last historical close price to connect with predictions
  const lastClosePrice = historicalData.close[historicalData.close.length - 1];
  
  // Create datasets for the chart
  const data = {
    labels: combinedDates,
    datasets: [
      {
        label: 'Historical Prices',
        data: [...historicalData.close, null], // Add null to avoid connecting to predictions
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        pointRadius: 2,
        pointHoverRadius: 5,
        borderWidth: 2,
        fill: true,
        tension: 0.1
      },
      {
        label: 'Volume',
        data: [...historicalData.volume.map(vol => vol / 1000000), ...Array(predictionDates.length).fill(null)],
        borderColor: 'rgba(75, 192, 192, 0.8)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        pointRadius: 0,
        borderWidth: 1,
        yAxisID: 'y1',
        fill: true,
        tension: 0.1
      },
      {
        label: 'Predictions',
        data: [...Array(dates.length - 1).fill(null), lastClosePrice, ...predictions],
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        pointRadius: 3,
        pointHoverRadius: 6,
        borderWidth: 2,
        borderDash: [5, 5],
        fill: false,
        tension: 0.1
      }
    ]
  };

  // Chart options
  const options = {
    responsive: true,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            
            if (context.parsed.y !== null) {
              if (context.dataset.label === 'Volume') {
                label += new Intl.NumberFormat('en-US', { 
                  style: 'decimal', 
                  maximumFractionDigits: 2 
                }).format(context.parsed.y) + 'M';
              } else {
                label += new Intl.NumberFormat('en-US', { 
                  style: 'currency', 
                  currency: 'USD' 
                }).format(context.parsed.y);
              }
            }
            return label;
          }
        }
      }
    },
    scales: {
      x: {
        ticks: {
          maxRotation: 45,
          minRotation: 45
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Price (USD)'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        grid: {
          drawOnChartArea: false,
        },
        title: {
          display: true,
          text: 'Volume (Millions)'
        }
      }
    }
  };

  return (
    <div className="stock-chart">
      <h3>Stock Price and Prediction Chart</h3>
      <div className="chart-container">
        <Line data={data} options={options} />
      </div>
      <div className="chart-legend">
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: 'rgba(54, 162, 235, 1)' }}></span>
          <span>Historical Prices</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: 'rgba(255, 99, 132, 1)' }}></span>
          <span>Predicted Prices</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: 'rgba(75, 192, 192, 0.8)' }}></span>
          <span>Volume</span>
        </div>
      </div>

      <style jsx>{`
        .stock-chart {
          margin-top: 20px;
        }
        
        .chart-container {
          height: 400px;
          margin-bottom: 20px;
        }
        
        .chart-legend {
          display: flex;
          justify-content: center;
          flex-wrap: wrap;
          gap: 20px;
          margin-top: 10px;
        }
        
        .legend-item {
          display: flex;
          align-items: center;
          gap: 5px;
        }
        
        .legend-color {
          width: 15px;
          height: 15px;
          border-radius: 3px;
        }
      `}</style>
    </div>
  );
};

export default StockChart; 