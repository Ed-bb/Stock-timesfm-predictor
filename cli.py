#!/usr/bin/env python3
"""Command-line interface for stock forecasting"""
import argparse
import sys
from datetime import datetime
import pandas as pd
from forecast_engine import StockForecaster
from visualize_forecast import plot_forecast, plot_comparison
from fetch_stock_data import DEFAULT_TICKER
from config import FORECAST_HORIZON, LOGS_DIR
import os

def format_forecast_table(forecast_result):
    ticker = forecast_result['ticker']
    current = forecast_result['current_price']
    forecast = forecast_result['forecast']
    quantiles = forecast_result['quantiles']
    dates = forecast_result['dates']
    
    print(f"\n{'='*70}")
    print(f"  {ticker} Stock Price Forecast")
    print(f"{'='*70}")
    print(f"  Current Price: ${current:.2f}")
    print(f"  Forecast Horizon: {len(forecast)} days")
    print(f"  Expected Change: {((forecast[-1] - current) / current * 100):+.2f}%")
    print(f"{'='*70}\n")
    
    print(f"{'Day':<5} {'Date':<12} {'Predicted':<12} {'10th %ile':<12} {'90th %ile':<12}")
    print(f"{'-'*70}")
    
    for i in range(len(forecast)):
        pred = forecast[i]
        q10 = quantiles[i, 0]
        q90 = quantiles[i, 8]
        date_str = dates[i].strftime("%Y-%m-%d")
        print(f"{i+1:<5} {date_str:<12} ${pred:<11.2f} ${q10:<11.2f} ${q90:<11.2f}")

def save_forecast_csv(forecast_result, ticker=None):
    if ticker is None:
        ticker = forecast_result['ticker']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(LOGS_DIR, f"{ticker}_forecast_{timestamp}.csv")
    
    data = {
        'date': forecast_result['dates'],
        'forecast': forecast_result['forecast'],
        'q10': forecast_result['quantiles'][:, 0],
        'q90': forecast_result['quantiles'][:, 8],
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Forecast saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(
        description=f"TimesFM Stock Forecaster (Default: {DEFAULT_TICKER})"
    )
    
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER, help='Stock ticker')
    parser.add_argument('--horizon', type=int, default=FORECAST_HORIZON, help='Forecast days')
    parser.add_argument('--compare', nargs='+', help='Compare stocks')
    parser.add_argument('--no-plot', action='store_true', help='Skip plots')
    parser.add_argument('--save-csv', action='store_true', help='Save to CSV')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None, help='Device')
    
    args = parser.parse_args()
    
    try:
        forecaster = StockForecaster(device=args.device)
        
        if args.compare:
            print(f"\nComparing: {', '.join(args.compare)}")
            results = forecaster.forecast_batch(args.compare, args.horizon)
            
            if results:
                for ticker, result in results.items():
                    format_forecast_table(result)
                    if args.save_csv:
                        save_forecast_csv(result)
                
                if not args.no_plot:
                    plot_comparison(results, save=True, show=False)
        else:
            print(f"\nForecasting {args.ticker}...")
            result = forecaster.forecast(args.ticker, args.horizon)
            format_forecast_table(result)
            
            if args.save_csv:
                save_forecast_csv(result)
            
            if not args.no_plot:
                plot_forecast(result, save=True, show=False)
    
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
