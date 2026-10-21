"""TimesFM forecasting engine"""
import numpy as np
import pandas as pd
from timesfm3 import TimesFM3Evaluator, ModelConfig
from fetch_stock_data import fetch_stock_data, DEFAULT_TICKER
from config import TIMESFM_CONFIG, FORECAST_HORIZON

class StockForecaster:
    """TimesFM forecaster for stock prices"""
    
    def __init__(self, device=None, checkpoint=None):
        if device is None:
            device = TIMESFM_CONFIG["device"]
        if checkpoint is None:
            checkpoint = TIMESFM_CONFIG["checkpoint_path"]
        
        print(f"Loading TimesFM 3.0 (device: {device})...")
        config = ModelConfig(
            checkpoint_path=checkpoint,
            per_core_batch_size=TIMESFM_CONFIG["per_core_batch_size"],
            device=device
        )
        self.forecaster = TimesFM3Evaluator(config)
        print("Model loaded successfully")
    
    @staticmethod
    def normalize_prices(prices):
        returns = np.diff(prices) / prices[:-1]
        return returns
    
    @staticmethod
    def denormalize(returns, last_price):
        prices = last_price * np.cumprod(1 + returns)
        return prices
    
    def forecast(self, ticker=None, horizon=None, use_returns=True):
        if ticker is None:
            ticker = DEFAULT_TICKER
        if horizon is None:
            horizon = FORECAST_HORIZON
        
        ticker = ticker.upper()
        print(f"\nForecasting {ticker}...")
        
        data = fetch_stock_data(ticker, save=True)
        prices = data['Close'].values.astype(np.float32)
        last_price = prices[-1]
        
        print(f"Current price: ${last_price:.2f}")
        
        if use_returns:
            input_data = self.normalize_prices(prices)
        else:
            input_data = prices
        
        outputs = list(self.forecaster.predict_batch(
            [input_data],
            horizon=horizon,
            return_quantiles=True
        ))
        
        forecast_returns = outputs[0].forecast
        quantiles_returns = outputs[0].quantiles
        
        if use_returns:
            forecast_prices = self.denormalize(forecast_returns, last_price)
            quantiles_prices = np.zeros((horizon, 9))
            for i in range(9):
                quantiles_prices[:, i] = self.denormalize(quantiles_returns[:, i], last_price)
        else:
            forecast_prices = forecast_returns
            quantiles_prices = quantiles_returns
        
        forecast_dates = pd.date_range(
            start=pd.Timestamp(data.index[-1]) + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        print(f"30-day prediction: ${forecast_prices[-1]:.2f}")
        print(f"Expected change: {((forecast_prices[-1] - last_price) / last_price * 100):+.2f}%")
        
        return {
            'ticker': ticker,
            'current_price': last_price,
            'forecast': forecast_prices,
            'quantiles': quantiles_prices,
            'dates': forecast_dates,
            'normalized': use_returns,
            'historical_prices': prices
        }
    
    def forecast_batch(self, tickers=None, horizon=None):
        if tickers is None:
            tickers = [DEFAULT_TICKER]
        
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.forecast(ticker, horizon)
            except Exception as e:
                print(f"Error forecasting {ticker}: {e}")
        
        return results
