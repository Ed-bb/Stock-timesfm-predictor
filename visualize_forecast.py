"""Visualization module for forecasts"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
from config import PLOT_DPI, PLOT_FIGSIZE, PLOTS_DIR

def plot_forecast(forecast_result, save=True, show=False):
    ticker = forecast_result['ticker']
    historical_prices = forecast_result['historical_prices']
    current_price = forecast_result['current_price']
    forecast_prices = forecast_result['forecast']
    quantiles = forecast_result['quantiles']
    forecast_dates = forecast_result['dates']
    
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    
    historical_dates = pd.date_range(
        end=forecast_dates[0] - pd.Timedelta(days=1),
        periods=len(historical_prices),
        freq='D'
    )
    plt.plot(historical_dates, historical_prices, 'b-', linewidth=2, label='Historical', alpha=0.8)
    plt.plot(forecast_dates, forecast_prices, 'r-', linewidth=2, label='Forecast', alpha=0.8)
    
    q10 = quantiles[:, 0]
    q90 = quantiles[:, 8]
    plt.fill_between(forecast_dates, q10, q90, alpha=0.2, color='red', label='80% CI')
    plt.axhline(y=current_price, color='green', linestyle='--', linewidth=1.5, label=f'Current: ${current_price:.2f}')
    
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Price ($)', fontsize=12, fontweight='bold')
    plt.title(f'{ticker} Forecast', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = None
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(PLOTS_DIR, f"{ticker}_forecast_{timestamp}.png")
        plt.savefig(filepath, dpi=PLOT_DPI)
        print(f"Plot saved to {filepath}")
    
    if show:
        plt.show()
    
    plt.close()
    return filepath

def plot_comparison(forecast_results, save=True, show=False):
    n_stocks = len(forecast_results)
    fig, axes = plt.subplots(n_stocks, 1, figsize=(14, 5 * n_stocks), dpi=PLOT_DPI)
    
    if n_stocks == 1:
        axes = [axes]
    
    for idx, (ticker, forecast) in enumerate(forecast_results.items()):
        ax = axes[idx]
        historical_prices = forecast['historical_prices']
        current_price = forecast['current_price']
        forecast_prices = forecast['forecast']
        quantiles = forecast['quantiles']
        forecast_dates = forecast['dates']
        
        historical_dates = pd.date_range(
            end=forecast_dates[0] - pd.Timedelta(days=1),
            periods=len(historical_prices),
            freq='D'
        )
        ax.plot(historical_dates, historical_prices, 'b-', linewidth=2, label='Historical', alpha=0.8)
        ax.plot(forecast_dates, forecast_prices, 'r-', linewidth=2, label='Forecast', alpha=0.8)
        
        q10 = quantiles[:, 0]
        q90 = quantiles[:, 8]
        ax.fill_between(forecast_dates, q10, q90, alpha=0.2, color='red')
        ax.axhline(y=current_price, color='green', linestyle='--', linewidth=1.5)
        
        change_pct = ((forecast_prices[-1] - current_price) / current_price) * 100
        ax.set_title(f'{ticker} | ${current_price:.2f} -> ${forecast_prices[-1]:.2f} ({change_pct:+.2f}%)', fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    
    filepath = None
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tickers = "_".join(forecast_results.keys())
        filepath = os.path.join(PLOTS_DIR, f"comparison_{tickers}_{timestamp}.png")
        plt.savefig(filepath, dpi=PLOT_DPI)
        print(f"Comparison saved to {filepath}")
    
    if show:
        plt.show()
    
    plt.close()
    return filepath
