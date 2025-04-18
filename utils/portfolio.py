import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from utils.auth import get_user_portfolio, save_user_portfolio
from utils.stock_data import get_stock_data, append_ns_suffix, remove_ns_suffix


def add_to_portfolio(ticker, shares, purchase_price):
    """
    Add a stock to the user's portfolio

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    shares : float
        Number of shares
    purchase_price : float
        Purchase price per share

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        if not st.session_state.logged_in:
            print("Error: User not logged in")
            return False

        username = st.session_state.username
        print(f"Adding/updating {ticker} to portfolio for user: {username}")

        # Get current portfolio
        portfolio = get_user_portfolio(username)
        print(f"Current portfolio has {len(portfolio)} items")

        # Standardize ticker format - make sure it's uppercase and trimmed
        ticker = ticker.strip().upper()

        # Check if ticker already exists
        for i, stock in enumerate(portfolio):
            stock_ticker = stock["ticker"].strip().upper()
            if stock_ticker == ticker:
                print(f"Updating existing position for {ticker}")
                # Update existing position
                current_shares = float(stock["shares"])
                current_cost = current_shares * float(stock["purchase_price"])
                new_shares = current_shares + float(shares)
                new_cost = current_cost + (float(shares) * float(purchase_price))
                new_avg_price = new_cost / new_shares

                portfolio[i]["shares"] = new_shares
                portfolio[i]["purchase_price"] = new_avg_price

                # Save portfolio
                save_result = save_user_portfolio(username, portfolio)
                print(f"Portfolio save result: {save_result}")
                return save_result

        # Add new position
        print(f"Adding new position for {ticker}")
        portfolio.append({
            "ticker": ticker,
            "shares": float(shares),
            "purchase_price": float(purchase_price)
        })

        # Save portfolio
        save_result = save_user_portfolio(username, portfolio)
        print(f"Portfolio save result: {save_result}")
        return save_result
    except Exception as e:
        print(f"Error in add_to_portfolio: {e}")
        return False


def remove_from_portfolio(ticker):
    """
    Remove a stock from the user's portfolio

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        if not st.session_state.logged_in:
            print("Error: User not logged in")
            return False

        username = st.session_state.username
        print(f"Removing {ticker} from portfolio for user: {username}")

        # Get current portfolio
        portfolio = get_user_portfolio(username)
        print(f"Current portfolio has {len(portfolio)} items")
        
        # Standardize ticker format
        ticker = ticker.strip().upper()

        # Find and remove ticker
        initial_length = len(portfolio)
        portfolio = [stock for stock in portfolio if stock["ticker"].strip().upper() != ticker]

        # Check if anything was removed
        if len(portfolio) == initial_length:
            print(f"Error: Ticker {ticker} not found in portfolio")
            return False

        print(f"Removed ticker {ticker}, new portfolio size: {len(portfolio)}")

        # Save portfolio
        save_result = save_user_portfolio(username, portfolio)
        print(f"Portfolio save result: {save_result}")
        return save_result
    except Exception as e:
        print(f"Error in remove_from_portfolio: {e}")
        return False


def update_portfolio_position(ticker, shares, purchase_price):
    """
    Update a position in the user's portfolio

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    shares : float
        New number of shares
    purchase_price : float
        New purchase price per share

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        if not st.session_state.logged_in:
            print("Error: User not logged in")
            return False

        username = st.session_state.username
        print(f"Updating position for {ticker} in portfolio for user: {username}")

        # Get current portfolio
        portfolio = get_user_portfolio(username)
        print(f"Current portfolio has {len(portfolio)} items")
        
        # Standardize ticker format
        ticker = ticker.strip().upper()

        # Find and update ticker
        found = False
        for i, stock in enumerate(portfolio):
            stock_ticker = stock["ticker"].strip().upper()
            if stock_ticker == ticker:
                print(f"Found ticker {ticker} at position {i}")
                # Convert values to ensure they're numeric
                portfolio[i]["shares"] = float(shares)
                portfolio[i]["purchase_price"] = float(purchase_price)
                found = True
                break

        if not found:
            print(f"Error: Ticker {ticker} not found in portfolio")
            return False

        print(f"Updated position for {ticker}")

        # Save portfolio
        save_result = save_user_portfolio(username, portfolio)
        print(f"Portfolio save result: {save_result}")
        return save_result
    except Exception as e:
        print(f"Error in update_portfolio_position: {e}")
        return False


def get_portfolio_summary():
    """
    Get a summary of the user's portfolio with current market values

    Returns:
    --------
    tuple
        (pandas.DataFrame, float, float, float): summary dataframe, total value, 
        total gain/loss, total gain/loss percentage
    """
    if not st.session_state.logged_in:
        return pd.DataFrame(), 0.0, 0.0, 0.0

    username = st.session_state.username

    # Get current portfolio
    portfolio = get_user_portfolio(username)

    if not portfolio:
        return pd.DataFrame(), 0.0, 0.0, 0.0

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(portfolio)

    # Get current prices using yfinance
    tickers = df["ticker"].tolist()

    # Print for debugging
    print(f"Portfolio tickers: {tickers}")

    try:
        #Clean and append .NS suffix if needed.  Handles both cases consistently.
        clean_tickers = [
            append_ns_suffix(t) if '.NS' not in t.upper() else t.upper()
            for t in tickers
        ]

        # Get stock data with proper NSE suffix handling
        data = get_stock_data(clean_tickers, period="5d")

        if data.empty:
            # If we couldn't get data, return the basic portfolio without current values
            print("Warning: Empty data returned from get_stock_data")
            return df, 0.0, 0.0, 0.0

        # Try to get the latest prices, using 'Close' as fallback if 'Adj Close' is not available
        try:
            # Check for MultiIndex in columns (which yfinance can sometimes return)
            if isinstance(data.columns, pd.MultiIndex):
                print("MultiIndex detected in data columns")
                # Try to get the price columns
                if ('Adj Close',) in data.columns or 'Adj Close' in data.columns:
                    price_column = 'Adj Close'
                else:
                    price_column = 'Close'
                
                # Extract the latest prices for each ticker
                latest_prices = data.iloc[-1]
                if isinstance(latest_prices.index, pd.MultiIndex):
                    print("MultiIndex in latest prices")
                    # Get the level with the price column
                    latest_prices = latest_prices.xs(price_column, level=0, axis=0)
                print(f"Using {price_column} for latest prices with MultiIndex")
            else:
                # Regular DataFrame
                if 'Adj Close' in data.columns:
                    latest_prices = data['Adj Close'].iloc[-1]
                    print("Using Adj Close for latest prices")
                else:
                    # Fallback to Close if Adj Close is not available
                    latest_prices = data['Close'].iloc[-1]
                    print("Using Close for latest prices (Adj Close not available)")

            # Print the latest prices for debugging
            print(f"Latest prices type: {type(latest_prices)}")
            print(f"Latest prices index: {latest_prices.index}")
            
            # Convert to dictionary for easier access
            if hasattr(latest_prices, 'to_dict'):
                latest_prices_dict = latest_prices.to_dict()
                print(f"Latest prices dict: {latest_prices_dict}")
            else:
                print(f"Latest prices not convertible to dict: {latest_prices}")
                latest_prices_dict = {}

            # Add current price to each position with robust error handling
            current_prices = []
            for ticker in df["ticker"]:
                try:
                    # Try multiple ways to match the ticker
                    original_ticker = ticker
                    ticker_with_ns = append_ns_suffix(ticker)
                    ticker_without_ns = remove_ns_suffix(ticker)
                    
                    # Method 1: Direct match in index
                    if ticker in latest_prices.index:
                        current_prices.append(latest_prices[ticker])
                        print(f"Found price for {ticker} (direct match)")
                    # Method 2: Match with .NS suffix
                    elif ticker_with_ns in latest_prices.index:
                        current_prices.append(latest_prices[ticker_with_ns])
                        print(f"Found price for {ticker} (with .NS suffix)")
                    # Method 3: Look for ticker without .NS suffix
                    elif ticker_without_ns in latest_prices.index:
                        current_prices.append(latest_prices[ticker_without_ns])
                        print(f"Found price for {ticker} (without .NS suffix)")
                    # Method 4: Find any match by removing .NS from all indices
                    else:
                        # Try to find a match by normalizing all ticker formats
                        matches = [
                            t for t in latest_prices.index
                            if remove_ns_suffix(t) == remove_ns_suffix(ticker)
                        ]
                        if matches:
                            current_prices.append(latest_prices[matches[0]])
                            print(f"Found price for {ticker} using normalized match: {matches[0]}")
                        else:
                            # Use purchase price as fallback
                            purchase_price = df[df["ticker"] == original_ticker]["purchase_price"].values[0]
                            current_prices.append(purchase_price)
                            print(f"Warning: Could not find price for {ticker}, using purchase price: {purchase_price}")
                except Exception as e:
                    # If any error occurs, use purchase price as fallback
                    # Make sure to use the original ticker from the portfolio
                    purchase_price = df[df["ticker"] == ticker]["purchase_price"].values[0]
                    current_prices.append(purchase_price)
                    print(f"Error getting price for {ticker}: {str(e)}. Using purchase price: {purchase_price}")

            df["current_price"] = current_prices
        except Exception as e:
            print(f"Error extracting latest prices: {e}")
            # Use purchase prices as fallback
            df["current_price"] = df["purchase_price"]
            print("Using purchase prices as fallback for current prices")

        # Ensure numeric types for calculations
        df["shares"] = pd.to_numeric(df["shares"], errors='coerce')
        df["purchase_price"] = pd.to_numeric(df["purchase_price"],
                                             errors='coerce')
        df["current_price"] = pd.to_numeric(df["current_price"],
                                            errors='coerce')

        # Calculate cost basis and current value
        df["cost_basis"] = df["shares"] * df["purchase_price"]
        df["current_value"] = df["shares"] * df["current_price"]

        # Calculate gain/loss with null handling
        df["gain_loss"] = df["current_value"] - df["cost_basis"]
        df["gain_loss_pct"] = df.apply(lambda x:
                                       (x["gain_loss"] / x["cost_basis"]) * 100
                                       if x["cost_basis"] > 0 else 0.0,
                                       axis=1)

        # Calculate total portfolio value and gain/loss
        total_value = df["current_value"].sum()
        total_cost = df["cost_basis"].sum()
        total_gain_loss = total_value - total_cost

        # Calculate total gain/loss percentage with zero handling
        total_gain_loss_pct = (total_gain_loss /
                               total_cost) * 100 if total_cost > 0 else 0.0

        return df, total_value, total_gain_loss, total_gain_loss_pct

    except Exception as e:
        print(f"Error in get_portfolio_summary: {e}")
        # Return basic portfolio information without pricing data
        return df, 0.0, 0.0, 0.0


def get_portfolio_allocations():
    """
    Get the current allocations of the portfolio

    Returns:
    --------
    pandas.DataFrame
        DataFrame with portfolio allocations
    """
    if not st.session_state.logged_in:
        return pd.DataFrame()

    username = st.session_state.username

    # Get portfolio summary
    summary, total_value, _, _ = get_portfolio_summary()

    if summary.empty or total_value == 0:
        return pd.DataFrame()

    # Calculate allocations
    allocations = pd.DataFrame({
        "ticker":
        summary["ticker"],
        "value":
        summary["current_value"],
        "allocation": (summary["current_value"] / total_value) * 100
    })

    # Sort by allocation in descending order
    allocations = allocations.sort_values("allocation", ascending=False)

    return allocations


def get_portfolio_returns_data():
    """Get the returns data and current weights for portfolio optimization"""
    if not st.session_state.logged_in:
        return pd.DataFrame(), None

    username = st.session_state.username
    portfolio = get_user_portfolio(username)

    if not portfolio:
        return pd.DataFrame(), None

    df = pd.DataFrame(portfolio)
    # Ensure tickers are standardized
    df['ticker'] = df['ticker'].apply(
        lambda x: remove_ns_suffix(x.strip().upper()))

    total_value = (df['shares'] * df['purchase_price']).sum()
    current_weights = (df['shares'] * df['purchase_price']) / total_value

    # Get returns data with clean tickers
    tickers = df["ticker"].tolist()
    try:
        #Clean and append .NS suffix if needed.  Handles both cases consistently.
        clean_tickers = [
            append_ns_suffix(t) if '.NS' not in t.upper() else t.upper()
            for t in tickers
        ]

        data = get_stock_data(clean_tickers, period="1y")
        if data.empty:
            return pd.DataFrame(), None

        if 'Adj Close' in data.columns:
            returns = data['Adj Close'].pct_change().dropna()
        else:
            returns = data['Close'].pct_change().dropna()

        return returns, current_weights.to_dict()
    except Exception as e:
        print(f"Error in get_portfolio_returns_data: {e}")
        return pd.DataFrame(), None
