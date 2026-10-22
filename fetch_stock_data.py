"""Stock data fetching module"""
import yfinance as yf
import pandas as pd
import os
from config import DATA_DIR, HISTORICAL_PERIOD, MIN_DATA_POINTS, DEFAULT_TICKER

def fetch_stock_data(ticker=None, period=None, save=True):
    """Fetch historical stock data from Yahoo Finance."""
    if ticker is None:
        ticker = DEFAULT_TICKER
    if period is None:
        period = HISTORICAL_PERIOD
    
    ticker = ticker.upper()
    print(f"Fetching {ticker} data for period: {period}")
    
    try:
        data = yf.download(ticker, period=period, progress=False)
        
        if data.empty:
            raise ValueError(f"No data for {ticker}")
        
        if len(data) < MIN_DATA_POINTS:
            raise ValueError(f"Insufficient data. Got {len(data)}, need {MIN_DATA_POINTS}")
        
        closes = data[['Close']].copy()
        closes.columns = ['Close']
        
        print(f"Fetched {len(closes)} data points")
        print(f"Price range: ${closes['Close'].min():.2f} - ${closes['Close'].max():.2f}")
        
        if save:
            filepath = os.path.join(DATA_DIR, f"{ticker}_historical.csv")
            closes.to_csv(filepath)
            print(f"Saved to {filepath}")
        
        return closes
    
    except Exception as e:
        raise ValueError(f"Error fetching {ticker}: {str(e)}")

def get_current_price(ticker=None):
    """Get current price for a stock."""
    if ticker is None:
        ticker = DEFAULT_TICKER
    
    try:
        data = yf.download(ticker, period="1d", progress=False)
        return float(data['Close'].iloc[-1])
    except Exception as e:
        print(f"Error: {e}")
        return None
