"""
Fetch Qantas (QAN.AX) share price data from 2010 to present
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
    """Fetch Qantas share price data from 2010 to present and ASX 200 index data"""

    print("Fetching Qantas (QAN.AX) share price data from 2010...")

    # Define date range - from 2010 to present
    end_date = datetime.now()
    start_date = datetime(2010, 1, 1)

    # Try different ticker symbols for Qantas
    ticker_symbols = ["QAN.AX", "QAN", "QANAS.AX"]
    ticker = None
    hist_data = None
    
    for symbol in ticker_symbols:
        print(f"Trying ticker symbol: {symbol}")
        try:
            ticker = yf.Ticker(symbol)
            print(f"  Created ticker object for {symbol}")
            
            # Test with a small date range first
            test_start = start_date
            test_end = start_date + timedelta(days=30)
            print(f"  Testing date range: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            
            test_data = ticker.history(start=test_start, end=test_end, interval="1d")
            print(f"  Test data shape: {test_data.shape}")
            
            if not test_data.empty:
                print(f"[SUCCESS] Found data with ticker: {symbol}")
                # Now get full data
                print(f"  Fetching full data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                hist_data = ticker.history(start=start_date, end=end_date, interval="1d")
                print(f"  Full data shape: {hist_data.shape}")
                break
            else:
                print(f"[FAILED] No data found for {symbol}")
        except Exception as e:
            print(f"[ERROR] Error with {symbol}: {e}")
            continue

    if hist_data is None or hist_data.empty:
        print("Error: No data retrieved for any Qantas ticker symbol.")
        print("Tried symbols:", ticker_symbols)
        print("This might be due to:")
        print("1. Internet connection issues")
        print("2. Yahoo Finance API issues")
        print("3. Ticker symbol changes")
        print("4. Market data restrictions")
        return None

    # Fetch ASX 200 index data
    print("Fetching ASX 200 (^AXJO) index data...")
    index_ticker = yf.Ticker("^AXJO")
    index_data = None
    
    try:
        index_data = index_ticker.history(start=start_date, end=end_date, interval="1d")
        if index_data.empty:
            print("Warning: No index data retrieved. Continuing without index data...")
            index_data = None
        else:
            print("[SUCCESS] ASX 200 index data retrieved successfully")
    except Exception as e:
        print(f"Warning: Error fetching index data: {e}. Continuing without index data...")
        index_data = None
    
    # Process data - merge Qantas and index data by date
    share_data = []
    for date, row in hist_data.iterrows():
        date_str = date.strftime("%Y-%m-%d")

        data_point = {
            "date": date_str,
            "open": round(float(row["Open"]), 3),
            "high": round(float(row["High"]), 3),
            "low": round(float(row["Low"]), 3),
            "close": round(float(row["Close"]), 3),
            "volume": int(row["Volume"]),
            "daily_change": round(float(row["Close"] - row["Open"]), 3),
            "daily_change_percent": round(((row["Close"] - row["Open"]) / row["Open"] * 100), 2)
        }

        # Add index data if available for this date
        if index_data is not None and date in index_data.index:
            index_row = index_data.loc[date]
            data_point["index_close"] = round(float(index_row["Close"]), 2)
            data_point["index_open"] = round(float(index_row["Open"]), 2)
            data_point["index_daily_change_percent"] = round(((index_row["Close"] - index_row["Open"]) / index_row["Open"] * 100), 2)

        share_data.append(data_point)
    
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

        # Add index data if present
        if 'index_close' in row and pd.notna(row['index_close']):
            data_point['index_close'] = row['index_close']
            data_point['index_open'] = row['index_open']
            data_point['index_daily_change_percent'] = row['index_daily_change_percent']

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

    # Add index statistics if available
    if index_data is not None:
        index_with_data = [d for d in share_data_with_ma if 'index_close' in d]
        if index_with_data:
            stats["index_start"] = index_with_data[0]['index_close']
            stats["index_end"] = index_with_data[-1]['index_close']
            stats["index_total_return"] = round((index_with_data[-1]['index_close'] - index_with_data[0]['index_close']) / index_with_data[0]['index_close'] * 100, 2)
            stats["index_days"] = len(index_with_data)
    
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
            "ticker": ticker_symbols[0] if ticker else "QAN.AX",
            "company": "Qantas Airways Limited",
            "exchange": "Australian Securities Exchange",
            "currency": "AUD",
            "index_ticker": "^AXJO",
            "index_name": "ASX 200",
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
    print(f"\nQANTAS ({ticker_symbols[0] if ticker else 'QAN.AX'}):")
    print(f"  Start price: ${stats['start_price']}")
    print(f"  End price: ${stats['end_price']}")
    print(f"  Total return: {stats['total_return']}%")
    print(f"  Min price: ${stats['min_price']}")
    print(f"  Max price: ${stats['max_price']}")
    print(f"  Significant drops (>5%): {len(significant_drops)}")

    if 'index_start' in stats:
        print(f"\nASX 200 INDEX (^AXJO):")
        print(f"  Start: {stats['index_start']}")
        print(f"  End: {stats['index_end']}")
        print(f"  Total return: {stats['index_total_return']}%")
        print(f"  Days with data: {stats['index_days']}")

    return output


if __name__ == "__main__":
    print("=" * 60)
    print("Qantas Share Price Data Fetcher")
    print("=" * 60)
    print("\nThis will fetch QAN.AX share price data from 2010 to present")
    print("Data source: Yahoo Finance")
    
    try:
        data = fetch_qantas_share_price()
        if data:
            print("\n[SUCCESS] Share price data fetched successfully!")
        else:
            print("\n[FAILED] Failed to fetch share price data")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease install yfinance:")
        print("  pip install yfinance pandas")