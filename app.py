import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import os
import urllib3
import json

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# App configuration
st.set_page_config(page_title="StockSense", page_icon="📈", layout="wide")

# Styling
st.markdown("""
<style>
    .main {padding-top: 0.5rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap;}
    div.stButton > button:first-child {background-color: #4CAF50; color:white;}
</style>
""", unsafe_allow_html=True)

# App header
st.title("📈 StockSense: Prediction & Analysis")
st.markdown("---")

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Alpha Vantage API Key", value="demo", type="password", help="Get your free API key at alphavantage.co")
    days_to_predict = st.slider("Days to Predict", 7, 30, 14)
    years_for_historical = st.slider("Years of Historical Data", 1, 5, 2)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("StockSense helps you analyze stocks and predict future prices using machine learning.")

# Main app logic
if not api_key:
    st.warning("Please enter your Alpha Vantage API key in the sidebar to continue.")
    st.stop()

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Function to get stock data
@st.cache_data(ttl=3600)
def get_stock_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': 'full',
        'apikey': api_key
    }
    try:
        st.sidebar.write(f"Fetching data for {symbol}...")
        response = requests.get(url, params=params, verify=False, timeout=10)
        st.sidebar.write(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            st.sidebar.error(f"API Error: {response.status_code} - {response.text}")
            return {"status": "error", "error": f"API Error: {response.status_code}"}
            
        data = response.json()
        
        # Check if there's an error message
        if "Error Message" in data:
            st.sidebar.error(f"API Error: {data['Error Message']}")
            return {"status": "error", "error": data["Error Message"]}
            
        if "Time Series (Daily)" not in data:
            st.sidebar.error("No time series data found in the response")
            return {"status": "error", "error": "No time series data found"}
            
        return {"status": "ok", "data": data}
    except Exception as e:
        st.sidebar.error(f"Error fetching stock data: {str(e)}")
        return {"status": "error", "error": str(e)}

# Function to search stocks
@st.cache_data(ttl=3600, show_spinner=False)
def search_stocks(query, api_key):
    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'SYMBOL_SEARCH',
        'keywords': query,
        'apikey': api_key
    }
    try:
        response = requests.get(url, params=params, verify=False, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "bestMatches" in data:
                return data["bestMatches"]
            else:
                st.sidebar.error(f"Search API Error: {json.dumps(data)}")
                return []
        else:
            st.sidebar.error(f"Search API Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.sidebar.error(f"Error searching stocks: {str(e)}")
        return []

# Function to predict stock prices
def predict_stock_prices(df, days_to_predict):
    # Prepare data
    df_copy = df.copy()
    df_copy['Prediction'] = df_copy['Close'].shift(-days_to_predict)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_copy[['Close']])
    
    # Create features and target
    X = np.array(range(len(scaled_data)))
    X = X.reshape(-1, 1)
    y = scaled_data
    
    # Train model
    model = LinearRegression()
    model.fit(X[:-days_to_predict], y[:-days_to_predict])
    
    # Predict future prices
    x_forecast = np.array(range(len(scaled_data), len(scaled_data) + days_to_predict))
    x_forecast = x_forecast.reshape(-1, 1)
    predicted_scaled = model.predict(x_forecast)
    
    # Inverse transform
    predicted_prices = scaler.inverse_transform(predicted_scaled)
    
    return predicted_prices.flatten()

# Stock search and selection
col1, col2 = st.columns([3, 1])
with col1:
    stock_query = st.text_input("Search for a stock symbol or company name", 
                               placeholder="Example: AAPL, Apple, MSFT, etc.")
with col2:
    analyze_btn = st.button("Analyze", use_container_width=True)

# Stock suggestions
if stock_query:
    suggestions = search_stocks(stock_query, api_key)
    if suggestions:
        options = [f"{s['1. symbol']} - {s['2. name']}" for s in suggestions]
        selected_option = st.selectbox("Select a stock", options)
        selected_symbol = selected_option.split(" - ")[0]
        
        # Add to history
        if selected_symbol not in [h['symbol'] for h in st.session_state.history]:
            st.session_state.history.append({
                'symbol': selected_symbol,
                'name': selected_option.split(" - ")[1] if " - " in selected_option else selected_symbol,
                'logo': ''  # Alpha Vantage doesn't provide logos
            })
            if len(st.session_state.history) > 5:
                st.session_state.history.pop(0)
    else:
        st.info("No stocks found. Try a different search term.")

# Recently viewed stocks
if st.session_state.history:
    st.markdown("### Recently Viewed")
    cols = st.columns(len(st.session_state.history))
    for i, stock in enumerate(st.session_state.history):
        with cols[i]:
            st.button(f"{stock['symbol']}", key=f"history_{i}")

# Tabs for different analyses
if 'selected_symbol' in locals() and analyze_btn:
    tabs = st.tabs(["📊 Price Analysis", "📈 Prediction", "💰 Return Calculator", "📰 News"])
    
    # Get stock data
    stock_data_response = get_stock_data(selected_symbol, api_key)
    
    if stock_data_response.get('status') == 'ok':
        # Process the data
        time_series = stock_data_response['data']['Time Series (Daily)']
        
        # Convert to DataFrame
        data_list = []
        for date, values in time_series.items():
            data_list.append({
                'Date': datetime.strptime(date, '%Y-%m-%d'),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': float(values['5. volume'])
            })
        
        df = pd.DataFrame(data_list)
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
        
        # Limit to last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_for_historical*365)
        df = df[df.index >= start_date]
        
        # Price Analysis Tab
        with tabs[0]:
            st.subheader(f"{selected_symbol} Price Analysis")
            
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ))
            
            # Moving averages
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20-day MA', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-day MA', line=dict(color='blue')))
            
            fig.update_layout(height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key statistics
            col1, col2, col3, col4 = st.columns(4)
            current_price = df['Close'].iloc[-1]
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", 
                         f"{(df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100:.2f}%")
            with col2:
                st.metric("52-Week High", f"${df['High'].max():.2f}")
            with col3:
                st.metric("52-Week Low", f"${df['Low'].min():.2f}")
            with col4:
                st.metric("Volatility", f"{df['Close'].pct_change().std() * 100:.2f}%")
        
        # Prediction Tab
        with tabs[1]:
            st.subheader(f"{selected_symbol} Price Prediction")
            
            # Predict future prices
            future_prices = predict_stock_prices(df, days_to_predict)
            
            # Create future dates
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': future_prices
            })
            pred_df.set_index('Date', inplace=True)
            
            # Plot predictions
            fig = go.Figure()
            
            # Historical prices
            fig.add_trace(go.Scatter(
                x=df.index[-30:],
                y=df['Close'].iloc[-30:],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Predicted prices
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['Predicted_Price'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='green', dash='dash')
            ))
            
            fig.update_layout(height=500, title=f"{days_to_predict}-Day Price Prediction")
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction summary
            current_price = df['Close'].iloc[-1]
            predicted_price = pred_df['Predicted_Price'].iloc[-1]
            price_change = predicted_price - current_price
            pct_change = (price_change / current_price) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    f"Predicted Price ({future_dates[-1].strftime('%Y-%m-%d')})",
                    f"${predicted_price:.2f}",
                    f"{pct_change:.2f}%"
                )
            with col2:
                prediction_direction = "Bullish 🟢" if pct_change > 0 else "Bearish 🔴"
                st.info(f"Prediction Outlook: {prediction_direction}")
            
            st.caption("Note: This prediction is based on a simple linear regression model and should not be used as financial advice.")
        
        # Return Calculator Tab
        with tabs[2]:
            st.subheader("Investment Return Calculator")
            
            # Add time period selection
            time_period = st.radio(
                "Investment Time Period",
                ["Historical", "Future Projection"],
                horizontal=True
            )
            
            if time_period == "Historical":
                col1, col2, col3 = st.columns(3)
                with col1:
                    investment_amount = st.number_input("Investment Amount ($)", min_value=100, value=1000, step=100)
                with col2:
                    # Get the earliest available date from the dataframe
                    min_date = df.index.min().date()
                    investment_date = st.date_input(
                        "Investment Date", 
                        value=min_date + timedelta(days=30),  # Default to 30 days after min date 
                        min_value=min_date,
                        max_value=end_date.date()
                    )
                with col3:
                    # Get the latest available date from the dataframe
                    max_date = df.index.max().date()
                    sell_date = st.date_input(
                        "Sell Date", 
                        value=max_date,
                        min_value=investment_date,
                        max_value=max_date
                    )
                
                # Calculate returns
                if investment_date and sell_date:
                    try:
                        # Ensure we get data on or after the investment date
                        investment_data = df.loc[df.index >= pd.Timestamp(investment_date)]
                        if investment_data.empty:
                            st.error("No data available for the selected investment date.")
                        else:
                            buy_price = investment_data.iloc[0]['Close']
                            
                            # Ensure we get data on or before the sell date
                            sell_data = df.loc[df.index <= pd.Timestamp(sell_date)]
                            if sell_data.empty:
                                st.error("No data available for the selected sell date.")
                            else:
                                sell_price = sell_data.iloc[-1]['Close']
                                
                                shares_bought = investment_amount / buy_price
                                final_value = shares_bought * sell_price
                                profit = final_value - investment_amount
                                roi = (profit / investment_amount) * 100
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Shares Purchased", f"{shares_bought:.2f}")
                                with col2:
                                    st.metric("Final Value", f"${final_value:.2f}")
                                with col3:
                                    st.metric("Return on Investment", f"{roi:.2f}%", f"${profit:.2f}")
                                
                                # Visualize investment growth
                                investment_period = df.loc[pd.Timestamp(investment_date):pd.Timestamp(sell_date)]
                                if not investment_period.empty:
                                    investment_period['Portfolio Value'] = investment_amount * (investment_period['Close'] / buy_price)
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=investment_period.index,
                                        y=investment_period['Portfolio Value'],
                                        fill='tozeroy',
                                        name='Portfolio Value'
                                    ))
                                    fig.update_layout(height=400, title="Portfolio Value Over Time")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Not enough data points between the selected dates to visualize portfolio growth.")
                    except Exception as e:
                        st.error(f"Error calculating returns: {str(e)}")
                        st.info("Try selecting different dates with available market data.")
            else:  # Future Projection
                col1, col2, col3 = st.columns(3)
                with col1:
                    investment_amount = st.number_input("Investment Amount ($)", min_value=100, value=1000, step=100)
                with col2:
                    # Today's date as the investment date
                    investment_date = st.date_input(
                        "Investment Date (Today)", 
                        value=datetime.now().date(),
                        disabled=True
                    )
                with col3:
                    # Future years
                    projection_years = st.slider("Projection Years", 1, 10, 5)
                    future_date = datetime.now() + timedelta(days=365 * projection_years)
                    st.write(f"Projected to: {future_date.strftime('%Y-%m-%d')}")
                
                # Get current price
                current_price = df.iloc[-1]['Close']
                
                # Calculate different growth scenarios
                conservative_rate = 0.05  # 5% annual growth
                moderate_rate = 0.08      # 8% annual growth
                aggressive_rate = 0.12    # 12% annual growth
                
                shares_bought = investment_amount / current_price
                
                # Calculate future values
                conservative_value = investment_amount * ((1 + conservative_rate) ** projection_years)
                moderate_value = investment_amount * ((1 + moderate_rate) ** projection_years)
                aggressive_value = investment_amount * ((1 + aggressive_rate) ** projection_years)
                
                # Display results
                st.subheader("Future Projection Scenarios")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Conservative (5% annual)", 
                        f"${conservative_value:.2f}", 
                        f"{(conservative_value - investment_amount) / investment_amount * 100:.1f}%"
                    )
                with col2:
                    st.metric(
                        "Moderate (8% annual)", 
                        f"${moderate_value:.2f}",
                        f"{(moderate_value - investment_amount) / investment_amount * 100:.1f}%"
                    )
                with col3:
                    st.metric(
                        "Aggressive (12% annual)", 
                        f"${aggressive_value:.2f}",
                        f"{(aggressive_value - investment_amount) / investment_amount * 100:.1f}%"
                    )
                
                # Projection chart
                years = list(range(projection_years + 1))
                conservative_values = [investment_amount * ((1 + conservative_rate) ** year) for year in years]
                moderate_values = [investment_amount * ((1 + moderate_rate) ** year) for year in years]
                aggressive_values = [investment_amount * ((1 + aggressive_rate) ** year) for year in years]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=years, 
                    y=conservative_values,
                    mode='lines+markers',
                    name='Conservative (5%)'
                ))
                fig.add_trace(go.Scatter(
                    x=years, 
                    y=moderate_values,
                    mode='lines+markers',
                    name='Moderate (8%)'
                ))
                fig.add_trace(go.Scatter(
                    x=years, 
                    y=aggressive_values,
                    mode='lines+markers',
                    name='Aggressive (12%)'
                ))
                
                fig.update_layout(
                    title="Projected Investment Growth by Year",
                    xaxis_title="Years",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Note about projection
                st.info("Note: These projections are based on hypothetical growth rates and do not account for market volatility, inflation, taxes, or fees. Actual returns may vary significantly.")

        # News Tab
        with tabs[3]:
            st.subheader(f"Latest News for {selected_symbol}")
            
            # Function to get news for a symbol
            @st.cache_data(ttl=1800)
            def get_news(symbol, api_key):
                try:
                    # Alpha Vantage news endpoint
                    news_url = "https://www.alphavantage.co/query"
                    news_params = {
                        'function': 'NEWS_SENTIMENT',
                        'tickers': symbol,
                        'apikey': api_key,
                        'limit': 10
                    }
                    
                    response = requests.get(news_url, params=news_params, verify=False, timeout=10)
                    if response.status_code == 200:
                        news_data = response.json()
                        if "feed" in news_data:
                            return news_data["feed"]
                        else:
                            st.warning("No news found or API limit reached.")
                            return []
                    else:
                        st.error(f"Failed to fetch news: {response.status_code}")
                        return []
                except Exception as e:
                    st.error(f"Error fetching news: {str(e)}")
                    return []
            
            news_items = get_news(selected_symbol, api_key)
            
            if news_items:
                # Display news articles
                for item in news_items:
                    st.markdown(f"### [{item.get('title', 'No Title')}]({item.get('url', '#')})")
                    st.markdown(f"*{item.get('source', 'Unknown Source')} - {item.get('time_published', 'Unknown Date')[:8]}*")
                    
                    # Truncate summary if too long
                    summary = item.get('summary', 'No summary available')
                    if len(summary) > 300:
                        summary = summary[:300] + "..."
                    st.markdown(summary)
                    
                    # Show sentiment if available
                    if "overall_sentiment_score" in item:
                        sentiment_score = float(item["overall_sentiment_score"])
                        sentiment_label = "Neutral"
                        
                        if sentiment_score > 0.25:
                            sentiment_label = "Positive"
                        elif sentiment_score < -0.25:
                            sentiment_label = "Negative"
                        
                        st.markdown(f"**Sentiment:** {sentiment_label} ({sentiment_score:.2f})")
                    
                    st.markdown("---")
            else:
                st.info(f"No recent news found for {selected_symbol} or daily API limit reached. Try again later or use your own API key.")
                st.markdown("Alpha Vantage's free tier has limited API calls. If you're seeing this message frequently, consider getting your own API key.")

    else:
        error_msg = stock_data_response.get('error', 'Unknown error')
        st.error(f"Failed to fetch stock data: {error_msg}")
        st.info("This could be due to:")
        st.info("1. Invalid API key (the default 'demo' key has limited usage)")
        st.info("2. Invalid stock symbol")
        st.info("3. Network connectivity issues")
        st.info("4. API rate limits (free tier is limited to 5 calls per minute and 500 calls per day)")
        
        # Display the request details for debugging
        st.expander("Debug Information").write({
            "Symbol": selected_symbol,
            "API Response": stock_data_response
        })

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Alpha Vantage API")
st.caption("Disclaimer: This app is for educational purposes only. Not financial advice.")