import streamlit as st
import json
import os
from passlib.hash import bcrypt
import requests
from datetime import datetime, timedelta

# Create a directory for storing user data
os.makedirs("data", exist_ok=True)
# Use absolute path for reliability
USERS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "users.json")
print(f"Users file path: {USERS_FILE}")

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if "username" not in st.session_state:
        st.session_state.username = ""
    
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

def hash_password(password):
    """Create a hash of the password"""
    return bcrypt.hash(password)

def verify_password(plain_password, hashed_password):
    """Verify a password against a hash"""
    return bcrypt.verify(plain_password, hashed_password)

def _load_users():
    """Load users from the JSON file"""
    if not os.path.exists(USERS_FILE):
        return {}
    
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def _save_users(users):
    """Save users to the JSON file"""
    try:
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        
        # Save the data
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)
        
        # Verify the data was written
        if os.path.exists(USERS_FILE):
            file_size = os.path.getsize(USERS_FILE)
            print(f"Successfully wrote {file_size} bytes to {USERS_FILE}")
            return True
        else:
            print(f"Error: File {USERS_FILE} was not created")
            return False
    except Exception as e:
        print(f"Error saving users data: {e}")
        return False

def register_user(username, password):
    """Register a new user"""
    users = _load_users()
    
    # Check if username already exists
    if username in users:
        return False
    
    # Hash the password
    hashed_password = hash_password(password)
    
    # Create user record
    users[username] = {
        "username": username,
        "password": hashed_password,
        "portfolio": []
    }
    
    # Save users
    _save_users(users)
    
    return True

def login_user(username, password):
    """Verify login credentials"""
    users = _load_users()
    
    # Check if username exists
    if username not in users:
        return False
    
    # Verify password
    if not verify_password(password, users[username]["password"]):
        return False
    
    return True

def check_login():
    """Check if user is logged in"""
    return st.session_state.logged_in

def save_user_portfolio(username, portfolio):
    """Save the user's portfolio"""
    try:
        # Load the current users data
        users = _load_users()
        
        print(f"Saving portfolio for user: {username}")
        print(f"Portfolio items: {len(portfolio)}")
        
        # Check if username exists
        if username not in users:
            print(f"Error: User '{username}' not found in users data")
            return False
        
        # Update portfolio
        users[username]["portfolio"] = portfolio
        
        # Save users and get result
        save_result = _save_users(users)
        
        # Double-check that the portfolio was updated
        if save_result:
            # Reload users to verify the change
            updated_users = _load_users()
            if username in updated_users:
                updated_portfolio = updated_users[username].get("portfolio", [])
                if len(updated_portfolio) == len(portfolio):
                    print(f"Portfolio successfully saved and verified for user: {username}")
                    return True
                else:
                    print(f"Error: Portfolio size mismatch after save. Expected: {len(portfolio)}, Actual: {len(updated_portfolio)}")
            else:
                print(f"Error: User '{username}' not found after saving")
        
        return save_result
    except Exception as e:
        print(f"Error in save_user_portfolio: {e}")
        return False

def get_user_portfolio(username):
    """Get the user's portfolio"""
    users = _load_users()
    
    # Check if username exists
    if username not in users:
        return []
    
    return users[username].get("portfolio", [])