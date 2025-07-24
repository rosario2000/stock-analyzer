# 🧠 Advanced Stock Analyzer - LSTM Enhanced Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced stock price prediction system that combines LSTM neural networks with statistical analysis to forecast potential upside movements. Features ensemble modeling, technical indicators, confidence intervals, and adaptive hybrid predictions for maximum accuracy and reliability.

## 🚀 Key Features

### Core Analysis
- Fetches real-time stock data from AlphaVantage API
- Calculates potential upside movements based on:
  - Historical volatility
  - Moving averages (5, 20, 50 day)
  - Current trend analysis
- Exports results to CSV format
- Rate limiting to respect API constraints
- Error handling for failed API calls

### 🧠 LSTM Neural Network Enhancement (NEW!)
- **Deep Learning Predictions**: Uses LSTM (Long Short-Term Memory) neural networks for time series forecasting
- **Multi-feature Analysis**: Incorporates open, high, low, close, and volume data
- **Hybrid Approach**: Combines LSTM predictions (70%) with statistical analysis (30%) for robust results
- **Enhanced Accuracy**: Learns temporal patterns and dependencies in stock price movements
- **Confidence Scoring**: Provides confidence metrics for LSTM predictions
- **Graceful Fallback**: Automatically falls back to statistical analysis if TensorFlow is unavailable

## Setup

### Prerequisites

- **Python 3.8-3.12** (for LSTM functionality) or Python 3.7+ (statistical mode only)
- AlphaVantage API key (included in the code: `R2UOE0T2GAU721CI`)

### Installation

1. Clone or download this project
2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Choose your installation option:**

   **Option A: Statistical Analysis Only (Faster, lighter)**
   ```bash
   pip install -r requirements.txt
   ```

   **Option B: Full LSTM Neural Network (Requires Python 3.8-3.12)**
   ```bash
   pip install -r requirements-lstm.txt
   ```

   **Note:** If you encounter TensorFlow installation issues on Python 3.13+, use Python 3.11 or 3.12, or use Option A for statistical analysis only.

## Usage

### Basic Usage

Make sure your virtual environment is activated first:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Then run the script with the default stock symbols:

```bash
python stock_analyzer.py
```

This will analyze these Indian stock symbols by default:
- HCLTECH.BSE
- TCS.BSE
- INFY.BSE
- WIPRO.BSE
- TECHM.BSE

### Custom Usage

You can modify the `stock_symbols` list in the `main()` function to analyze different stocks:

```python
stock_symbols = [
    "HCLTECH.BSE",
    "RELIANCE.BSE",
    "HDFC.BSE",
    # Add your symbols here
]
```

### Programmatic Usage

```python
from stock_analyzer import StockAnalyzer

# Initialize analyzer
analyzer = StockAnalyzer()

# Analyze specific stocks
symbols = ["HCLTECH.BSE", "TCS.BSE"]
results = analyzer.analyze_stocks(symbols)

# Save to CSV
csv_file = analyzer.save_to_csv(results, "my_analysis.csv")
```

## Output

The program generates a CSV file with the following columns:

### Core Columns
| Column | Description |
|--------|-------------|
| Symbol | Stock symbol |
| Status | Analysis status (Success/Error) |
| Latest Close Price | Most recent closing price |
| Latest Date | Date of the latest data |
| Upside 1 Day (%) | Potential upside movement for next day |
| Upside 1 Week (%) | Potential upside movement for next week |
| Upside 1 Month (%) | Potential upside movement for next month |
| Current Trend | Bullish/Bearish/Neutral trend indicator |
| Volatility 20D (%) | 20-day historical volatility |

### 🧠 LSTM Enhancement Columns (when enabled)
| Column | Description |
|--------|-------------|
| Prediction Method | "Statistical" or "LSTM+Statistical" |
| LSTM 1D Target | LSTM predicted price for next day |
| LSTM 1W Max | Maximum LSTM predicted price in next 7 days |
| LSTM 1M Max | Maximum LSTM predicted price in next 30 days |
| LSTM Confidence (%) | Confidence score of LSTM predictions (0-100) |

## How It Works

### 🧠 LSTM Neural Network Approach (When Available)

The LSTM enhancement uses advanced deep learning for time series prediction:

1. **Data Preparation**: Creates sequences of 60 days of historical data (open, high, low, close, volume)
2. **Neural Network Architecture**: 
   - 3 LSTM layers with 50 neurons each
   - Dropout layers (20%) for regularization
   - Dense layers for final price prediction
3. **Training Process**: 
   - Trains on historical patterns with early stopping
   - Uses 80% data for training, 20% for validation
   - Adam optimizer with learning rate 0.001
4. **Prediction Generation**: 
   - Predicts next 30 days of prices
   - Calculates upside potential from predicted maximums
5. **Hybrid Combination**: 70% LSTM + 30% Statistical analysis

### Statistical Analysis (Always Available)

Traditional technical analysis approach:

1. **Historical Volatility**: Standard deviation of daily returns
2. **Trend Analysis**: Comparison with moving averages (5, 20, 50 day)
3. **Time-based Scaling**: Different multipliers for day/week/month predictions
4. **Trend Multiplier**: Adjusts predictions based on bullish/bearish trends

### Calculation Logic

#### LSTM Mode (Enhanced)
- **1 Day Upside**: Weighted combination of LSTM prediction and statistical volatility
- **1 Week Upside**: Maximum LSTM predicted price in next 7 days vs current price
- **1 Month Upside**: Maximum LSTM predicted price in next 30 days vs current price

#### Statistical Mode (Fallback)
- **1 Day Upside**: Based on 5-day volatility with trend adjustment
- **1 Week Upside**: 20-day volatility × 2.5 with trend adjustment
- **1 Month Upside**: 20-day volatility × 4.0 with trend adjustment

### Trend Detection

- **Bullish**: Current price > 5-day MA and 20-day MA (multiplier: 1.2)
- **Bearish**: Current price < 5-day MA and 20-day MA (multiplier: 0.8)
- **Neutral**: Mixed signals (multiplier: 1.0)
- **LSTM Trend**: Additional trend classification from neural network predictions

## API Rate Limits

The script includes automatic rate limiting:
- 12-second delay between API calls
- Stays within AlphaVantage's 5 calls per minute limit (free tier)

## Error Handling

The script handles various error scenarios:
- Network connectivity issues
- API rate limiting
- Invalid stock symbols
- Missing data
- JSON parsing errors

## Sample Output

### With LSTM Neural Network (Option B Installation)
```
Stock Analysis Tool with LSTM Neural Network
============================================================
✓ LSTM neural network enabled for enhanced predictions
Analyzing 5 stocks...
This may take a while due to API rate limits and LSTM training...

Processing HCLTECH.BSE (1/5)
Fetching data for HCLTECH.BSE...
  🧠 Training LSTM model for HCLTECH.BSE...
✓ HCLTECH.BSE: 🧠 1D=3.24%, 1W=7.89%, 1M=15.78%

Processing TCS.BSE (2/5)
Fetching data for TCS.BSE...
  🧠 Training LSTM model for TCS.BSE...
✓ TCS.BSE: 🧠 1D=2.89%, 1W=6.72%, 1M=13.44%

...

Results saved to: stock_analysis_20241201_143022.csv

==================================================
ANALYSIS SUMMARY
==================================================
Successfully analyzed: 5 stocks
Failed to analyze: 0 stocks

Top potential upside movements (1 month):
  HCLTECH.BSE: 15.78% (Bullish (LSTM-Bullish))
  TCS.BSE: 13.44% (Neutral (LSTM-Bullish))
  ...
```

### Statistical Mode Only (Option A Installation or TensorFlow unavailable)
```
Stock Analysis Tool with LSTM Neural Network
============================================================
Warning: TensorFlow not available. LSTM predictions will be disabled.
Analyzing 5 stocks...
This may take a while due to API rate limits...

Processing HCLTECH.BSE (1/5)
Fetching data for HCLTECH.BSE...
✓ HCLTECH.BSE: 📊 1D=2.34%, 1W=5.89%, 1M=11.78%

...
```

## Troubleshooting

### SSL Certificate Issues

If you encounter SSL certificate verification errors (common in corporate networks), you have a few options:

1. **Use a VPN or different network** (recommended)
2. **Contact your IT department** to whitelist the AlphaVantage domain
3. **As a last resort**, you can temporarily disable SSL verification by modifying the `fetch_stock_data` method:
   ```python
   response = requests.get(self.base_url, params=params, verify=False, timeout=30)
   ```
   ⚠️ **Warning**: Only use this for testing on trusted networks.

### API Rate Limits

- The free AlphaVantage API allows 5 calls per minute
- The script automatically adds 12-second delays between calls
- For faster processing, consider upgrading to a paid API plan

### Common Issues

- **"No time series data found"**: The stock symbol might be incorrect or not supported
- **"API Rate limit"**: Wait a few minutes before running the script again
- **Network timeouts**: Check your internet connection and try again

## 📁 Project Structure

```
advanced-stock-analyzer/
├── stock_analyzer.py       # Main analysis engine with LSTM + Statistical hybrid
├── example_usage.py        # Example usage patterns and demonstrations  
├── requirements.txt        # Basic dependencies (statistical mode)
├── requirements-lstm.txt   # Full dependencies (LSTM mode)
├── README.md              # This file
├── .gitignore            # Git ignore patterns
└── output/               # Generated CSV analysis files (ignored by git)
```

## 📊 Sample Output

The system generates comprehensive CSV reports with metrics including:
- **Ensemble Agreement**: 96.6% (model consensus)
- **Confidence Intervals**: 95% and 68% prediction ranges
- **Risk Adjustment**: Sharpe ratio-based risk considerations
- **Adaptive Weights**: Dynamic LSTM/Statistical balancing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see below for details.

## ⚠️ Disclaimer

**Important**: This tool is for educational and research purposes only. The upside movement predictions are based on historical data and technical analysis. They should NOT be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment decisions.

## 🙏 Acknowledgments

- [AlphaVantage](https://www.alphavantage.co/) for providing the stock market API
- [TensorFlow](https://tensorflow.org/) for the machine learning framework
- The open-source community for the foundational libraries

---

**MIT License**

Copyright (c) 2025 Advanced Stock Analyzer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 