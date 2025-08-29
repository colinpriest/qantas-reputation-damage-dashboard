"""
Fetch Qantas (QAN.AX) share price data for the past 5 years
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Not critical for this script

def fetch_qantas_share_price():
    """Fetch 5 years of Qantas share price data"""
    
    print("Fetching Qantas (QAN.AX) share price data...")
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)
    
    # Fetch data using yfinance
    ticker = yf.Ticker("QAN.AX")
    
    # Get historical data
    hist_data = ticker.history(start=start_date, end=end_date, interval="1d")
    
    if hist_data.empty:
        print("Error: No data retrieved. Check ticker symbol or internet connection.")
        return None
    
    # Process data
    share_data = []
    for date, row in hist_data.iterrows():
        share_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 3),
            "high": round(float(row["High"]), 3),
            "low": round(float(row["Low"]), 3),
            "close": round(float(row["Close"]), 3),
            "volume": int(row["Volume"]),
            "daily_change": round(float(row["Close"] - row["Open"]), 3),
            "daily_change_percent": round(((row["Close"] - row["Open"]) / row["Open"] * 100), 2)
        })
    
    # Calculate moving averages
    df = pd.DataFrame(share_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Add moving averages
    df['ma_7'] = df['close'].rolling(window=7).mean().round(3)
    df['ma_30'] = df['close'].rolling(window=30).mean().round(3)
    df['ma_90'] = df['close'].rolling(window=90).mean().round(3)
    
    # Convert back to list of dicts
    df = df.reset_index()
    share_data_with_ma = []
    for _, row in df.iterrows():
        data_point = {
            "date": row['date'].strftime("%Y-%m-%d"),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume'],
            "daily_change": row['daily_change'],
            "daily_change_percent": row['daily_change_percent']
        }
        
        # Add moving averages if not NaN
        if pd.notna(row['ma_7']):
            data_point['ma_7'] = row['ma_7']
        if pd.notna(row['ma_30']):
            data_point['ma_30'] = row['ma_30']
        if pd.notna(row['ma_90']):
            data_point['ma_90'] = row['ma_90']
            
        share_data_with_ma.append(data_point)
    
    # Calculate statistics
    stats = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "total_days": len(share_data),
        "start_price": share_data[0]['close'] if share_data else 0,
        "end_price": share_data[-1]['close'] if share_data else 0,
        "min_price": min(d['low'] for d in share_data) if share_data else 0,
        "max_price": max(d['high'] for d in share_data) if share_data else 0,
        "avg_price": round(sum(d['close'] for d in share_data) / len(share_data), 3) if share_data else 0,
        "total_return": round((share_data[-1]['close'] - share_data[0]['close']) / share_data[0]['close'] * 100, 2) if share_data else 0
    }
    
    # Identify significant drops (> 5% in a day)
    significant_drops = []
    for data_point in share_data:
        if data_point['daily_change_percent'] < -5:
            significant_drops.append({
                "date": data_point['date'],
                "change_percent": data_point['daily_change_percent'],
                "close": data_point['close']
            })
    
    # Save to JSON
    output = {
        "metadata": {
            "ticker": "QAN.AX",
            "company": "Qantas Airways Limited",
            "exchange": "Australian Securities Exchange",
            "currency": "AUD",
            "fetched_at": datetime.now().isoformat()
        },
        "statistics": stats,
        "significant_drops": sorted(significant_drops, key=lambda x: x['change_percent'])[:20],
        "data": share_data_with_ma
    }
    
    # Save to file
    output_file = "qantas_share_price_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nShare price data saved to {output_file}")
    print(f"Period: {stats['start_date']} to {stats['end_date']}")
    print(f"Total trading days: {stats['total_days']}")
    print(f"Start price: ${stats['start_price']}")
    print(f"End price: ${stats['end_price']}")
    print(f"Total return: {stats['total_return']}%")
    print(f"Min price: ${stats['min_price']}")
    print(f"Max price: ${stats['max_price']}")
    print(f"Significant drops (>5%): {len(significant_drops)}")
    
    return output


if __name__ == "__main__":
    print("=" * 60)
    print("Qantas Share Price Data Fetcher")
    print("=" * 60)
    print("\nThis will fetch 5 years of QAN.AX share price data")
    print("Data source: Yahoo Finance")
    
    try:
        data = fetch_qantas_share_price()
        if data:
            print("\n✓ Share price data fetched successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease install yfinance:")
        print("  pip install yfinance pandas")