# StockSense: Stock Price Prediction & Analysis App

A streamlined, feature-rich stock analysis application built with Streamlit and Alpha Vantage API. This app provides stock price prediction, technical analysis, and return calculation in a clean, user-friendly interface.

## Features

- **Smart Stock Search**: Auto-suggests stocks as you type
- **Price Prediction**: Machine learning-based price forecasting
- **Technical Analysis**: Candlestick charts with moving averages
- **Return Calculator**: Calculate potential investment returns
- **Recently Viewed**: Quick access to previously analyzed stocks

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application with all functionality
- `requirements.txt`: Project dependencies

## API Key

This application uses the Alpha Vantage API for stock data. Get your free API key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

The app comes with a demo API key, but it has limited usage. For the best experience, get your own free API key.

## Deployment to Streamlit Cloud

1. Create a GitHub repository with this code
2. Sign up at [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set up your Alpha Vantage API key as a secret
5. Deploy with a single click

## Usage

1. Enter your Alpha Vantage API key in the sidebar (or use the default demo key)
2. Search for a stock by symbol or company name
3. Select from the suggestions
4. Click "Analyze" to view detailed analysis
5. Navigate between tabs to explore different features

## Notes on SSL Verification

This app disables SSL verification to handle potential certificate issues. For production use, consider implementing proper certificate handling.