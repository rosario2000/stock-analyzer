import streamlit as st

st.title("🚀 Stock Analyzer")

# Test basic functionality
symbol = st.text_input("Enter Stock Symbol:", "KPITTECH.BSE")

if st.button("Test Basic Functionality"):
    st.write("Testing import...")
    try:
        from stock_analyzer import StockAnalyzer
        st.success("✅ StockAnalyzer imported successfully!")
        
        # Test creation
        analyzer = StockAnalyzer(use_lstm=False)
        st.success("✅ Analyzer created!")
        
        if symbol:
            st.write(f"Analyzing {symbol}...")
            results = analyzer.analyze_stocks([symbol])
            
            if results and results[0]['status'] == 'Success':
                result = results[0]
                st.success("✅ Analysis complete!")
                
                # Display results
                st.metric("Current Price", f"₹{result['latest_close']:,.2f}")
                st.metric("1-Day Upside", f"{result['upside_1_day']:.2f}%")
                st.metric("1-Week Upside", f"{result['upside_1_week']:.2f}%")
                st.metric("1-Month Upside", f"{result['upside_1_month']:.2f}%")
                st.metric("1-Year Upside", f"{result['upside_1_year']:.2f}%")
                
            else:
                st.error("❌ Analysis failed")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.code(str(e))

st.info("👆 Click the button above to test the stock analyzer!") 