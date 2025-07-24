#!/usr/bin/env python3
"""
Example usage of the Stock Analyzer

This script demonstrates how to use the StockAnalyzer class
with custom stock symbols and settings.
"""

from stock_analyzer import StockAnalyzer

def analyze_custom_stocks():
    """
    Example function to analyze custom stock symbols
    """
    # Define your custom stock symbols
    custom_symbols = [
        "HCLTECH.BSE",   # HCL Technologies
        "TCS.BSE",       # Tata Consultancy Services
        "RELIANCE.BSE",  # Reliance Industries
        "HDFC.BSE",      # HDFC Bank
        "ICICIBANK.BSE", # ICICI Bank
    ]
    
    print("Custom Stock Analysis")
    print("=" * 40)
    print(f"Analyzing {len(custom_symbols)} custom stocks...")
    
    # Initialize the analyzer
    analyzer = StockAnalyzer()
    
    # Analyze the stocks
    results = analyzer.analyze_stocks(custom_symbols)
    
    # Save results with custom filename
    csv_filename = analyzer.save_to_csv(results, "custom_stock_analysis.csv")
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 80)
    
    for result in results:
        if result['status'] == 'Success':
            print(f"""
Stock: {result['symbol']}
Latest Price: ₹{result['latest_close']:.2f} (as of {result['latest_date']})
Trend: {result['current_trend']}
Potential Upside:
  - Next Day: {result['upside_1_day']}%
  - Next Week: {result['upside_1_week']}%
  - Next Month: {result['upside_1_month']}%
Volatility (20D): {result['volatility_20d']}%
{'-' * 80}""")
        else:
            print(f"""
Stock: {result['symbol']}
Status: {result['status']}
{'-' * 80}""")
    
    return results

def analyze_single_stock(symbol: str):
    """
    Example function to analyze a single stock
    """
    print(f"Analyzing single stock: {symbol}")
    
    analyzer = StockAnalyzer()
    results = analyzer.analyze_stocks([symbol])
    
    if results and results[0]['status'] == 'Success':
        result = results[0]
        print(f"""
Analysis for {symbol}:
- Current Price: ₹{result['latest_close']:.2f}
- Trend: {result['current_trend']}
- Upside Potential (1 month): {result['upside_1_month']}%
""")
    else:
        print(f"Failed to analyze {symbol}")
    
    return results

if __name__ == "__main__":
    print("Stock Analyzer - Example Usage")
    print("=" * 50)
    
    # Example 1: Analyze multiple custom stocks
    print("\n1. Analyzing multiple custom stocks...")
    analyze_custom_stocks()
    
    # Example 2: Analyze a single stock
    print("\n2. Analyzing a single stock...")
    analyze_single_stock("HCLTECH.BSE")
    
    print("\nExample completed! Check the generated CSV files for detailed results.") 