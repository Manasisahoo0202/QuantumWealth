import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

# NSE Popular Stocks List
NSE_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", 
    "SBIN", "BHARTIARTL", "BAJFINANCE", "KOTAKBANK", "LT", "AXISBANK", 
    "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO", 
    "HCLTECH", "TATAMOTORS", "INDUSINDBK", "ADANIENT", "NTPC", "ONGC", 
    "POWERGRID", "JSWSTEEL", "BAJAJFINSV", "TATASTEEL", "ADANIPORTS", 
    "TECHM", "HINDALCO", "DIVISLAB", "DRREDDY", "NESTLEIND", "CIPLA", 
    "APOLLOHOSP", "COALINDIA", "GRASIM", "UPL", "LTIM", "BAJAJ-AUTO", 
    "BPCL", "EICHERMOT", "HEROMOTOCO", "TATACONSUM", "M&M", "BRITANNIA", 
    "INDIGO", "SBILIFE", "HDFCLIFE"
]

def append_ns_suffix(ticker):
    """
    Append .NS suffix to Indian stock tickers for Yahoo Finance

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol

    Returns:
    --------
    str
        Ticker with .NS suffix if needed
    """
    # If it already has .NS suffix, return as is
    if ticker.endswith('.NS'):
        return ticker
    return f"{ticker}.NS"

def remove_ns_suffix(ticker):
    """
    Remove .NS suffix from ticker for display purposes

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol with potential .NS suffix

    Returns:
    --------
    str
        Ticker without .NS suffix
    """
    return ticker.replace('.NS', '')

def get_stock_data(ticker_symbols, period="1y"):
    """
    Fetch historical stock data for given ticker symbols

    Parameters:
    -----------
    ticker_symbols : list
        List of stock ticker symbols
    period : str
        Time period for historical data (e.g., '1y', '6mo', '3mo')

    Returns:
    --------
    pandas.DataFrame
        DataFrame with stock price data
    """
    if not ticker_symbols:
        return pd.DataFrame()

    try:
        # Standardize ticker format and add .NS suffix if not present
        ns_tickers = []
        for ticker in ticker_symbols:
            ticker = ticker.upper()
            if not ticker.endswith('.NS'):
                ns_tickers.append(f"{ticker}.NS")
            else:
                ns_tickers.append(ticker)

        # Create a string of tickers for yfinance
        tickers_str = " ".join(ns_tickers)

        # Print debug info
        print(f"Fetching data for tickers: {tickers_str}")

        # Get data for all tickers at once
        data = yf.download(tickers_str, period=period, progress=False, auto_adjust=False)

        # Print columns for debugging
        print(f"Data columns: {data.columns}")

        # If only one ticker, the structure is different
        if len(ticker_symbols) == 1:
            if 'Adj Close' not in data.columns:
                # If Adj Close is missing, use Close instead
                new_cols = ['Adj Close' if col == 'Close' else col for col in data.columns]
                data.columns = new_cols
            # Create MultiIndex for consistency
            data.columns = pd.MultiIndex.from_product([data.columns, [ns_tickers[0]]])

        # Rename columns to remove .NS suffix for consistency
        if len(ticker_symbols) > 1:
            # Check if we have Adj Close column
            if 'Adj Close' not in data.columns.levels[0]:
                # If missing, clone the Close column as Adj Close
                close_data = data['Close'].copy()
                data['Adj Close'] = close_data

            # For multi-level columns, need to modify the second level
            new_cols = data.columns.levels[1].str.replace('.NS', '')
            data.columns = data.columns.set_levels(new_cols, level=1)

        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def calculate_returns(data):
    """
    Calculate daily returns from price data

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock price data

    Returns:
    --------
    pandas.DataFrame
        DataFrame with daily returns
    """
    if data.empty:
        return pd.DataFrame()

    try:
        # Try to get the adjusted close prices or fall back to close prices
        if 'Adj Close' in data.columns:
            price_data = data['Adj Close']
            print("Using Adj Close for returns calculation")
        else:
            # Fallback to Close if Adj Close is not available
            price_data = data['Close']
            print("Using Close for returns calculation (Adj Close not available)")

        # Fill any NA values before calculating percentage change
        # First check for and handle any NA values to prevent errors
        if price_data.isna().any().any():
            print("Warning: Found NA values in price data, filling with forward fill method")
            price_data = price_data.fillna(method='ffill').fillna(method='bfill')

        # Calculate daily returns with explicit fill_method=None to avoid the warning/error
        returns = price_data.pct_change(fill_method=None).dropna()

        return returns
    except Exception as e:
        print(f"Error calculating returns: {e}")
        return pd.DataFrame()



def validate_tickers(ticker_symbols):
    """
    Validate if the provided ticker symbols exist

    Parameters:
    -----------
    ticker_symbols : list
        List of stock ticker symbols

    Returns:
    --------
    tuple
        (valid_tickers, invalid_tickers)
    """
    valid_tickers = []
    invalid_tickers = []

    print(f"Validating tickers: {ticker_symbols}")
    
    # First, pre-validate by checking against NSE_STOCKS
    for ticker in ticker_symbols:
        # Standardize and clean ticker first
        ticker = ticker.strip().upper()
        # Remove .NS if present
        clean_ticker = remove_ns_suffix(ticker)
        
        # If the ticker is in our pre-vetted NSE_STOCKS list, consider it valid
        # This helps with tickers that might have API validation issues but are known to be valid
        if clean_ticker in NSE_STOCKS:
            print(f"Ticker {clean_ticker} pre-validated as part of NSE_STOCKS list")
            valid_tickers.append(clean_ticker)
            continue
            
        # Add .NS for validation
        ns_ticker = f"{clean_ticker}.NS"

        try:
            ticker_data = yf.Ticker(ns_ticker)
            info = ticker_data.info
            
            # Check for regularMarketPrice (newer Yahoo Finance API)
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                valid_tickers.append(clean_ticker)  # Store without .NS
            # Also check for currentPrice (older Yahoo Finance API)
            elif 'currentPrice' in info and info['currentPrice'] is not None:
                valid_tickers.append(clean_ticker)
            # Also check for last price as fallback
            elif 'previousClose' in info and info['previousClose'] is not None:
                valid_tickers.append(clean_ticker)
            else:
                print(f"Ticker {clean_ticker} has no valid price information")
                invalid_tickers.append(clean_ticker)
        except Exception as e:
            print(f"Error validating ticker {clean_ticker}: {str(e)}")
            invalid_tickers.append(clean_ticker)

    print(f"Validation results - Valid: {valid_tickers}, Invalid: {invalid_tickers}")
    return valid_tickers, invalid_tickers


def find_high_potential_stocks(current_tickers, period="1y", top_n=5):
    """
    Find high-potential stocks that could complement the current portfolio

    Parameters:
    -----------
    current_tickers : list
        List of current stock tickers in the portfolio
    period : str
        Time period for historical data (e.g., '1y', '6mo', '3mo')
    top_n : int
        Number of top stocks to return

    Returns:
    --------
    dict
        Dictionary with high-potential stocks info
    """
    print(f"Finding high potential stocks that complement: {current_tickers}")
    print(f"Using period: {period}, top_n: {top_n}")
    # Load all NSE symbols
    all_nse_symbols = load_nse_symbols()
    
    # Select a subset of popular NSE stocks not in the current portfolio
    # We'll use NSE_STOCKS as a starting point since they're more liquid and popular
    candidate_tickers = [ticker for ticker in NSE_STOCKS if ticker not in current_tickers]
    
    # If we have too few candidates, add more from the full NSE list
    if len(candidate_tickers) < top_n * 3:
        additional_candidates = [ticker for ticker in all_nse_symbols.keys() 
                               if ticker not in current_tickers and ticker not in candidate_tickers]
        # Limit to a reasonable number to avoid lengthy processing
        candidate_tickers.extend(additional_candidates[:50])
    
    # Take a random sample of candidates to diversify suggestions
    # and limit processing time
    if len(candidate_tickers) > 20:
        import random
        # Use seed for consistent results between runs
        random.seed(42)
        candidate_sample = random.sample(candidate_tickers, min(20, len(candidate_tickers)))
    else:
        candidate_sample = candidate_tickers
    
    # Validate tickers to make sure they exist and are tradable
    valid_candidates, _ = validate_tickers(candidate_sample)
    
    if not valid_candidates:
        return {
            "high_return_stocks": [],
            "error": "No valid candidate stocks found"
        }
    
    try:
        # Get historical data for candidates
        candidate_data = get_stock_data(valid_candidates, period=period)
        
        if candidate_data.empty:
            return {
                "high_return_stocks": [],
                "error": "Failed to fetch historical data for candidate stocks"
            }
            
        # Calculate returns
        candidate_returns = calculate_returns(candidate_data)
        
        if candidate_returns.empty:
            return {
                "high_return_stocks": [],
                "error": "Failed to calculate returns for candidate stocks"
            }
        
        # Calculate metrics for each candidate
        stock_metrics = {}
        
        for ticker in valid_candidates:
            # Look for this ticker in the returns data
            # Need to handle potential suffix issues
            col_matches = [col for col in candidate_returns.columns if ticker in col]
            if not col_matches:
                continue
                
            ticker_col = col_matches[0]
            
            # Calculate annualized return, volatility, and Sharpe ratio
            annual_return = candidate_returns[ticker_col].mean() * 252
            volatility = candidate_returns[ticker_col].std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            # Recent momentum (last month vs. total period)
            recent_returns = candidate_returns[ticker_col].iloc[-20:].mean() * 252  # Last ~1 month
            momentum = recent_returns / annual_return if annual_return != 0 else 0
            
            stock_metrics[ticker] = {
                "ticker": ticker,
                "company_name": all_nse_symbols.get(ticker, ticker),
                "annual_return": annual_return * 100,  # Convert to percentage
                "volatility": volatility * 100,  # Convert to percentage
                "sharpe_ratio": sharpe,
                "momentum": momentum
            }
        
        # Sort by a combination of return and Sharpe ratio to find high-potential stocks
        # Different sorts for different metrics
        high_return_stocks = sorted(
            [m for m in stock_metrics.values() if m["annual_return"] > 0],
            key=lambda x: x["annual_return"], 
            reverse=True
        )[:top_n]
        
        high_sharpe_stocks = sorted(
            [m for m in stock_metrics.values() if m["sharpe_ratio"] > 0],
            key=lambda x: x["sharpe_ratio"], 
            reverse=True
        )[:top_n]
        
        high_momentum_stocks = sorted(
            [m for m in stock_metrics.values() if m["momentum"] > 1.0],  # Momentum > 1 means improving returns
            key=lambda x: x["momentum"], 
            reverse=True
        )[:top_n]
        
        # Combine results with weights for final ranking
        all_candidates = {}
        
        # Add scores for each category (return, sharpe, momentum)
        for i, stock in enumerate(high_return_stocks):
            ticker = stock["ticker"]
            if ticker not in all_candidates:
                all_candidates[ticker] = stock.copy()
                all_candidates[ticker]["score"] = 0
            all_candidates[ticker]["score"] += (top_n - i) * 0.5  # Return score
            
        for i, stock in enumerate(high_sharpe_stocks):
            ticker = stock["ticker"]
            if ticker not in all_candidates:
                all_candidates[ticker] = stock.copy()
                all_candidates[ticker]["score"] = 0
            all_candidates[ticker]["score"] += (top_n - i) * 0.3  # Sharpe score
            
        for i, stock in enumerate(high_momentum_stocks):
            ticker = stock["ticker"]
            if ticker not in all_candidates:
                all_candidates[ticker] = stock.copy()
                all_candidates[ticker]["score"] = 0
            all_candidates[ticker]["score"] += (top_n - i) * 0.2  # Momentum score
        
        # Get final sorted list of stocks
        final_candidates = sorted(
            all_candidates.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_n]
        
        return {
            "high_return_stocks": final_candidates,
            "all_candidates_count": len(valid_candidates)
        }
        
    except Exception as e:
        import traceback
        print(f"Error finding high-potential stocks: {str(e)}")
        print(traceback.format_exc())
        return {
            "high_return_stocks": [],
            "error": f"Error analyzing candidate stocks: {str(e)}"
        }

def load_nse_symbols():
    """
    Load NSE symbols from CSV file

    Returns:
    --------
    dict
        Dictionary with ticker symbols as keys and company names as values
    """
    import os
    import csv
    symbols_dict = {}

    # Define the path to the NSE symbols CSV file
    csv_path = os.path.join('data', 'market', 'nse_symbols.csv')

    try:
        with open(csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header row
            for row in csv_reader:
                if len(row) >= 2:
                    symbol = row[0].strip()
                    company_name = row[1].strip()
                    series = row[2].strip() if len(row) > 2 else ""

                    # Only add EQ series stocks by default (equity stocks)
                    if series == "EQ" or symbol in NSE_STOCKS:
                        symbols_dict[symbol] = company_name

        return symbols_dict
    except Exception as e:
        print(f"Error loading NSE symbols: {e}")
        # Return the default NSE_STOCKS as a dictionary if file not found
        return {stock: stock for stock in NSE_STOCKS}

