import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for classy, minimalistic design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        padding-top: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .input-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .metric-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .chart-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 2rem 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid #d1d5db;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_analyzer():
    """Load the stock analyzer with caching for better performance"""
    try:
        from stock_analyzer import StockAnalyzer
        return StockAnalyzer
    except Exception as e:
        st.error(f"Failed to load analyzer: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-title">Advanced Stock Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">AI-powered stock predictions with LSTM neural networks</p>', unsafe_allow_html=True)
    
    # Check if analyzer can be loaded
    StockAnalyzer = load_analyzer()
    if StockAnalyzer is None:
        st.error("Unable to load the stock analyzer. Please check the deployment configuration.")
        return
    
    # Input section
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        symbol = st.text_input("Stock Symbol", placeholder="Enter stock symbol (e.g., KPITTECH.BSE)", label_visibility="collapsed")
        
        if not symbol:
            symbol = "KPITTECH.BSE"
        
        # Analysis mode selection
        mode = st.selectbox("Analysis Mode", ["Statistical Only", "LSTM + Statistical"], index=0, label_visibility="collapsed")
        
        analyze_button = st.button("🔍 Analyze Stock", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis section
    if analyze_button and symbol:
        analyze_stock(symbol.upper().strip(), mode, StockAnalyzer)
    elif not analyze_button:
        show_placeholder()

def show_placeholder():
    """Show placeholder content when no analysis is running"""
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #9ca3af;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
        <h3 style="color: #6b7280; font-weight: 500;">Ready to analyze your stock</h3>
        <p style="color: #9ca3af;">Enter a stock symbol above and click "Analyze Stock" to get started</p>
    </div>
    """, unsafe_allow_html=True)

def analyze_stock(symbol, mode, StockAnalyzer):
    """Perform comprehensive stock analysis"""
    
    with st.spinner('🔄 Analyzing stock data...'):
        try:
            # Initialize analyzer based on mode
            use_lstm = mode == "LSTM + Statistical"
            analyzer = StockAnalyzer(use_lstm=use_lstm)
            
            # Perform analysis
            results = analyzer.analyze_stocks([symbol])
            
            if not results or results[0]['status'] != 'Success':
                st.error(f"❌ Unable to analyze {symbol}. Please verify the symbol and try again.")
                return
            
            result = results[0]
            
            # Display results
            display_results(result, analyzer, use_lstm)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            with st.expander("Error Details"):
                st.code(str(e))

def display_results(result, analyzer, has_lstm):
    """Display comprehensive analysis results"""
    
    # Key metrics row
    st.markdown("### 📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">₹{result['latest_close']:,.2f}</div>
            <div class="metric-label">Current Price</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        trend_color = "#10b981" if "Bullish" in result['current_trend'] else "#ef4444" if "Bearish" in result['current_trend'] else "#f59e0b"
        trend_text = result['current_trend'].split('(')[0].strip()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: {trend_color};">{trend_text}</div>
            <div class="metric-label">Market Trend</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{result['volatility_20d']:.1f}%</div>
            <div class="metric-label">Volatility (20D)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        method_text = "AI + Stats" if has_lstm else "Statistical"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{method_text}</div>
            <div class="metric-label">Method Used</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Price chart
    if has_lstm:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        create_price_chart(result, analyzer)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Upside potential metrics
    st.markdown("### 📈 Upside Potential")
    
    col1, col2, col3, col4 = st.columns(4)
    
    upside_data = [
        ("1 Day", result['upside_1_day'], col1),
        ("1 Week", result['upside_1_week'], col2),
        ("1 Month", result['upside_1_month'], col3),
        ("1 Year", result['upside_1_year'], col4)
    ]
    
    for period, value, col in upside_data:
        color = "#10b981" if value >= 5 else "#f59e0b" if value >= 2 else "#6b7280"
        with col:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="color: {color};">+{value:.2f}%</div>
                <div class="metric-label">{period}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional info
    st.markdown("### ℹ️ Analysis Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Last Updated:** {result['latest_date']}")
        st.markdown(f"**Prediction Method:** {result.get('prediction_method', 'Statistical')}")
    
    with col2:
        if has_lstm and result.get('lstm_confidence'):
            st.markdown(f"**AI Confidence:** {result['lstm_confidence']}%")
        st.markdown(f"**Data Source:** AlphaVantage API")

def create_price_chart(result, analyzer):
    """Create price chart with predictions"""
    try:
        symbol = result['symbol']
        historical_data = analyzer.fetch_stock_data(symbol)
        
        if historical_data is None or historical_data.empty:
            st.warning("Unable to fetch historical data for charting")
            return
        
        # Use last 30 days for faster loading
        historical_data = historical_data.tail(30).copy()
        historical_data.index = pd.to_datetime(historical_data.index)
        
        current_price = result['latest_close']
        last_date = pd.to_datetime(result['latest_date'])
        
        fig = go.Figure()
        
        # Historical price line
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#6366f1', width=2),
            hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>'
        ))
        
        # Current price marker
        fig.add_trace(go.Scatter(
            x=[last_date],
            y=[current_price],
            mode='markers',
            name='Current Price',
            marker=dict(size=10, color='#dc2626', line=dict(width=2, color='white')),
            hovertemplate='<b>Current</b><br>Price: ₹%{y:,.2f}<extra></extra>'
        ))
        
        # Predictions if available
        if result.get('lstm_1d_target') and result.get('lstm_1d_target') != 'N/A':
            future_dates = [
                last_date + timedelta(days=1),
                last_date + timedelta(days=7),
                last_date + timedelta(days=30)
            ]
            
            future_prices = [
                result.get('lstm_1d_target', current_price),
                result.get('lstm_1w_max', current_price),
                result.get('lstm_1m_max', current_price)
            ]
            
            # Filter valid predictions
            valid_predictions = [(d, p) for d, p in zip(future_dates, future_prices) if p != 'N/A' and p is not None]
            
            if valid_predictions:
                pred_dates = [last_date] + [d for d, _ in valid_predictions]
                pred_prices = [current_price] + [float(p) for _, p in valid_predictions]
                
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=pred_prices,
                    mode='lines+markers',
                    name='AI Predictions',
                    line=dict(color='#10b981', width=2, dash='dot'),
                    marker=dict(size=6, color='#10b981'),
                    hovertemplate='<b>Prediction</b><br>Date: %{x}<br>Price: ₹%{y:,.2f}<extra></extra>'
                ))
        
        fig.update_layout(
            title=f'<b>{symbol}</b> - Price Chart & AI Predictions',
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=400,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family='Inter'),
            margin=dict(t=60, b=40, l=40, r=40)
        )
        
        fig.update_xaxis(showgrid=True, gridcolor='#f3f4f6', gridwidth=1)
        fig.update_yaxis(showgrid=True, gridcolor='#f3f4f6', gridwidth=1, tickformat='₹,.0f')
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Unable to create price chart: {str(e)}")

if __name__ == "__main__":
    main() 