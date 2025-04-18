from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import json
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


# For QPSO algorithm, importing from the existing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.qpso import QPSO

# Security configurations
SECRET_KEY = "supersecretkey123"  # In production, use a secure key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Setup password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize FastAPI app
app = FastAPI(title="QuantumWealth API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User storage (in-memory for now, would use a database in production)
users_db = {}

# Models
class User(BaseModel):
    username: str
    hashed_password: str
    portfolio: List[Dict[str, Any]] = []

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class PortfolioItem(BaseModel):
    ticker: str
    shares: float
    purchase_price: float

class StockOptimizationRequest(BaseModel):
    tickers: List[str]
    risk_appetite: str = "Balanced"
    period: str = "1y"

# Security functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    if username in users_db:
        return User(**users_db[username])
    return None

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Stock data functions
def get_stock_data(ticker_symbols, period="1y"):
    """Fetch historical stock data for given ticker symbols"""
    if not ticker_symbols:
        return pd.DataFrame()
    
    try:
        # Create a string of tickers for yfinance
        tickers_str = " ".join(ticker_symbols)
        
        # Get data for all tickers at once
        data = yf.download(tickers_str, period=period)
        
        # If only one ticker, the structure is different
        if len(ticker_symbols) == 1:
            data.columns = pd.MultiIndex.from_product([['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], ticker_symbols])
        
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def calculate_returns(data):
    """Calculate daily returns from price data"""
    if data.empty:
        return pd.DataFrame()
    
    # Get the adjusted close prices
    adj_close = data['Adj Close']
    
    # Calculate daily returns
    returns = adj_close.pct_change().dropna()
    
    return returns

def validate_tickers(ticker_symbols):
    """Validate if the provided ticker symbols exist"""
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in ticker_symbols:
        try:
            ticker = ticker.strip().upper()
            ticker_data = yf.Ticker(ticker)
            
            # Try to get the info to see if the ticker is valid
            info = ticker_data.info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except:
            invalid_tickers.append(ticker)
    
    return valid_tickers, invalid_tickers

def optimize_portfolio(returns_data, risk_appetite='Balanced'):
    """Optimize portfolio allocation using QPSO"""
    if returns_data.empty or returns_data.shape[1] < 2:
        return None
    
    # Set weights based on risk appetite
    if risk_appetite == 'Conservative':
        w_return = 0.3
        w_risk = 0.7
    elif risk_appetite == 'Balanced':
        w_return = 0.5
        w_risk = 0.5
    else:  # Aggressive
        w_return = 0.7
        w_risk = 0.3
    
    # Create and run the optimizer
    optimizer = QPSO(returns_data, w_return=w_return, w_risk=w_risk)
    results = optimizer.optimize()
    
    # Format the results
    weights_df = pd.DataFrame({
        'Asset': returns_data.columns,
        'Weight': results['weights']
    })
    
    # Sort by weight in descending order
    weights_df = weights_df.sort_values('Weight', ascending=False)
    
    formatted_results = {
        'weights': weights_df.to_dict(orient='records'),
        'expected_annual_return': results['expected_annual_return'] * 100,  # Convert to percentage
        'expected_volatility': results['expected_volatility'] * 100,  # Convert to percentage
        'sharpe_ratio': results['sharpe_ratio']
    }
    
    return formatted_results

# API Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    if user.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Validate username and password
    if len(user.username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be at least 3 characters"
        )
    if len(user.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters"
        )
    
    hashed_password = get_password_hash(user.password)
    users_db[user.username] = {
        "username": user.username,
        "hashed_password": hashed_password,
        "portfolio": []
    }
    return {"detail": "User created successfully"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username, "portfolio": current_user.portfolio}

@app.post("/portfolio/add")
async def add_to_portfolio(item: PortfolioItem, current_user: User = Depends(get_current_user)):
    # Validate ticker
    valid_tickers, _ = validate_tickers([item.ticker])
    if item.ticker not in valid_tickers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ticker symbol: {item.ticker}"
        )
    
    # Check if ticker already exists in the portfolio
    user_data = users_db[current_user.username]
    updated = False
    
    for i, stock in enumerate(user_data["portfolio"]):
        if stock["ticker"] == item.ticker:
            # Update existing position
            current_shares = stock["shares"]
            current_cost = stock["shares"] * stock["purchase_price"]
            new_shares = current_shares + item.shares
            new_cost = current_cost + (item.shares * item.purchase_price)
            new_avg_price = new_cost / new_shares
            
            user_data["portfolio"][i]["shares"] = new_shares
            user_data["portfolio"][i]["purchase_price"] = new_avg_price
            updated = True
            break
    
    if not updated:
        # Add new position
        user_data["portfolio"].append({
            "ticker": item.ticker,
            "shares": item.shares,
            "purchase_price": item.purchase_price
        })
    
    return {"detail": "Portfolio updated successfully"}

@app.delete("/portfolio/{ticker}")
async def remove_from_portfolio(ticker: str, current_user: User = Depends(get_current_user)):
    user_data = users_db[current_user.username]
    initial_length = len(user_data["portfolio"])
    
    user_data["portfolio"] = [
        stock for stock in user_data["portfolio"] if stock["ticker"] != ticker
    ]
    
    if len(user_data["portfolio"]) < initial_length:
        return {"detail": "Stock removed from portfolio"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock not found in portfolio"
        )

@app.put("/portfolio/{ticker}")
async def update_portfolio_position(ticker: str, item: PortfolioItem, current_user: User = Depends(get_current_user)):
    if ticker != item.ticker:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticker in path must match ticker in request body"
        )
    
    user_data = users_db[current_user.username]
    updated = False
    
    for i, stock in enumerate(user_data["portfolio"]):
        if stock["ticker"] == ticker:
            user_data["portfolio"][i]["shares"] = item.shares
            user_data["portfolio"][i]["purchase_price"] = item.purchase_price
            updated = True
            break
    
    if updated:
        return {"detail": "Portfolio position updated"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock not found in portfolio"
        )

@app.get("/portfolio/summary")
async def get_portfolio_summary(current_user: User = Depends(get_current_user)):
    user_data = users_db[current_user.username]
    portfolio = user_data["portfolio"]
    
    if not portfolio:
        return {
            "summary": [],
            "total_value": 0.0,
            "total_gain_loss": 0.0,
            "total_gain_loss_pct": 0.0
        }
    
    # Extract tickers
    tickers = [stock["ticker"] for stock in portfolio]
    
    # Get current stock data
    data = get_stock_data(tickers, period="5d")
    
    if data.empty:
        return {
            "summary": [],
            "total_value": 0.0,
            "total_gain_loss": 0.0,
            "total_gain_loss_pct": 0.0
        }
    
    # Get the latest prices
    latest_prices = data['Adj Close'].iloc[-1]
    
    # Create portfolio summary
    portfolio_summary = []
    total_current_value = 0
    total_cost_basis = 0
    
    for stock in portfolio:
        ticker = stock["ticker"]
        shares = stock["shares"]
        purchase_price = stock["purchase_price"]
        
        current_price = latest_prices[ticker] if ticker in latest_prices.index else 0
        cost_basis = shares * purchase_price
        current_value = shares * current_price
        gain_loss = current_value - cost_basis
        gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
        
        portfolio_summary.append({
            "ticker": ticker,
            "shares": shares,
            "purchase_price": purchase_price,
            "current_price": current_price,
            "cost_basis": cost_basis,
            "current_value": current_value,
            "gain_loss": gain_loss,
            "gain_loss_pct": gain_loss_pct
        })
        
        total_current_value += current_value
        total_cost_basis += cost_basis
    
    # Calculate total portfolio gain/loss
    total_gain_loss = total_current_value - total_cost_basis
    total_gain_loss_pct = (total_gain_loss / total_cost_basis) * 100 if total_cost_basis > 0 else 0
    
    return {
        "summary": portfolio_summary,
        "total_value": total_current_value,
        "total_gain_loss": total_gain_loss,
        "total_gain_loss_pct": total_gain_loss_pct
    }

@app.get("/portfolio/allocations")
async def get_portfolio_allocations(current_user: User = Depends(get_current_user)):
    portfolio_data = await get_portfolio_summary(current_user)
    summary = portfolio_data["summary"]
    total_value = portfolio_data["total_value"]
    
    if not summary or total_value == 0:
        return {"allocations": []}
    
    # Calculate allocations
    allocations = []
    for item in summary:
        allocations.append({
            "ticker": item["ticker"],
            "value": item["current_value"],
            "allocation": (item["current_value"] / total_value) * 100
        })
    
    return {"allocations": sorted(allocations, key=lambda x: x["allocation"], reverse=True)}

@app.post("/optimize")
async def optimize_new_portfolio(request: StockOptimizationRequest):
    # Validate tickers
    valid_tickers, invalid_tickers = validate_tickers(request.tickers)
    
    if invalid_tickers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ticker symbols: {', '.join(invalid_tickers)}"
        )
    
    if len(valid_tickers) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 valid tickers are needed for portfolio optimization"
        )
    
    # Fetch historical data
    data = get_stock_data(valid_tickers, period=request.period)
    
    if data.empty:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve historical data"
        )
    
    # Calculate returns
    returns = calculate_returns(data)
    
    if returns.empty:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to calculate returns from historical data"
        )
    
    # Run optimization
    optimization_results = optimize_portfolio(returns, request.risk_appetite)
    
    if not optimization_results:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Optimization failed"
        )
    
    return optimization_results

@app.post("/optimize/portfolio")
async def optimize_existing_portfolio(
    request: Optional[StockOptimizationRequest] = None,
    current_user: User = Depends(get_current_user)
):
    user_data = users_db[current_user.username]
    portfolio = user_data["portfolio"]
    
    if not portfolio or len(portfolio) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You need at least 2 stocks in your portfolio to run optimization"
        )
    
    # Extract tickers
    tickers = [stock["ticker"] for stock in portfolio]
    
    # Create a request if not provided
    if not request:
        request = StockOptimizationRequest(
            tickers=tickers,
            risk_appetite="Balanced",
            period="1y"
        )
    
    # Get historical data
    data = get_stock_data(tickers, period=request.period)
    
    if data.empty:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve historical data for your portfolio"
        )
    
    # Calculate returns
    returns = calculate_returns(data)
    
    if returns.empty:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to calculate returns from historical data"
        )
    
    # Run optimization
    optimization_results = optimize_portfolio(returns, request.risk_appetite)
    
    if not optimization_results:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Optimization failed"
        )
    
    return optimization_results

# For development
if __name__ == "__main__":
    import sys
    uvicorn.run(app, host="0.0.0.0", port=8000)