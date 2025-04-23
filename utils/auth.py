import streamlit as st
import json
import os
import uuid
import time
from passlib.hash import bcrypt
import requests
from datetime import datetime, timedelta
from collections import defaultdict

# Create a directory for storing user data
os.makedirs("data", exist_ok=True)
# Use absolute path for reliability
USERS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
    "users.json")
print(f"Users file path: {USERS_FILE}")

# Session constants
SESSION_TOKEN_PARAM = "token"
SESSION_PAGE_PARAM = "page"
SESSION_EXPIRY_DAYS = 7  # How long the session token is valid


def get_query_params():
    """Get query parameters from the current URL"""
    try:
        # First try to use the newer st.query_params API
        return st.query_params.to_dict()
    except:
        try:
            # Then try to use st.context.headers (which replaces _get_websocket_headers)
            if hasattr(st, 'context') and hasattr(st.context, 'headers'):
                uri = st.context.headers.get("URI", "")
                # Get query part from URI
                if "?" in uri:
                    query_string = uri.split("?")[1]
                    params = {}
                    for param in query_string.split("&"):
                        if "=" in param:
                            key, value = param.split("=", 1)
                            params[key] = value
                    return params
            # Last resort, try to use experimental get query params
            return st.experimental_get_query_params()
        except:
            # If all methods fail, return empty dictionary
            return {}


def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    # Basic session state initialization
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "username" not in st.session_state:
        st.session_state.username = ""

    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = datetime.now()

    if "last_activity" not in st.session_state:
        st.session_state.last_activity = datetime.now()
        
    # Check for session token in URL if not already logged in
    if not st.session_state.logged_in:
        params = get_query_params()
        token = params.get(SESSION_TOKEN_PARAM)
        
        if token:
            # Try to validate token and restore session
            username = validate_session_token(token)
            if username:
                # Restore session data
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.session_start_time = datetime.now()
                st.session_state.last_activity = datetime.now()
                
                # Restore the page if it's in the URL parameters
                page_param = params.get(SESSION_PAGE_PARAM)
                if page_param in ["dashboard", "optimization"]:
                    st.session_state.page = page_param
                else:
                    # Default to dashboard if no valid page parameter
                    st.session_state.page = "dashboard"
                    
                print(f"Session restored for user: {username} on page: {st.session_state.page}")


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


def validate_password(password):
    """Validate password complexity"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain uppercase letters"
    if not any(c.islower() for c in password):
        return False, "Password must contain lowercase letters"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain numbers"
    if not any(c in "!@#$%^&*" for c in password):
        return False, "Password must contain special characters (!@#$%^&*)"
    return True, "Password is valid"


def register_user(username, password):
    """Register a new user"""
    users = _load_users()

    # Check if username already exists
    if username in users:
        return False, "Username already exists"

    # Validate password
    is_valid, message = validate_password(password)
    if not is_valid:
        return False, message

    # Hash the password
    hashed_password = hash_password(password)

    # Create user record
    users[username] = {
        "username": username,
        "password": hashed_password,
        "portfolio": [],
        "needs_password_update": False,  # New passwords already meet requirements
        "failed_login_attempts": []
    }

    # Save users
    if _save_users(users):
        return True, "Registration successful"
    else:
        return False, "Error saving user data"


def generate_session_token(username):
    """Generate a unique session token for the user"""
    # Create a unique token based on username, timestamp, and random UUID
    unique_string = f"{username}:{datetime.now().timestamp()}:{uuid.uuid4()}"
    # Hash it for security
    token = bcrypt.hash(unique_string)
    # Store token expiry time (in seconds since epoch)
    expiry = int(time.time() + (SESSION_EXPIRY_DAYS * 24 * 60 * 60))
    
    return token, expiry


def validate_session_token(token):
    """
    Validate a session token and return the username if valid
    
    Returns:
        str: Username if token is valid, None otherwise
    """
    users = _load_users()
    current_time = int(time.time())
    
    # Check each user for matching token
    for username, user_data in users.items():
        if "session_token" in user_data and user_data["session_token"] == token:
            # Check if token has expired
            if "token_expiry" in user_data and user_data["token_expiry"] > current_time:
                # Valid token, update expiry time
                users[username]["token_expiry"] = int(time.time() + (SESSION_EXPIRY_DAYS * 24 * 60 * 60))
                _save_users(users)
                return username
            else:
                # Token expired, clear it
                users[username].pop("session_token", None)
                users[username].pop("token_expiry", None)
                _save_users(users)
    
    return None


def logout_user(username):
    """
    Log out a user by invalidating their session token
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        users = _load_users()
        if username in users:
            # Remove session token
            users[username].pop("session_token", None)
            users[username].pop("token_expiry", None)
            _save_users(users)
            
            # Remove token from URL - use the built-in query params API
            try:
                params = st.query_params.to_dict()
                if SESSION_TOKEN_PARAM in params:
                    # Clear the token
                    st.query_params.clear()
                    # Add back all other params except the token
                    for key, value in params.items():
                        if key != SESSION_TOKEN_PARAM:
                            st.query_params[key] = value
            except:
                # For compatibility with older Streamlit versions
                try:
                    params = st.experimental_get_query_params()
                    if SESSION_TOKEN_PARAM in params:
                        del params[SESSION_TOKEN_PARAM]
                        st.experimental_set_query_params(**params)
                except:
                    print("Could not modify URL query parameters")
                
            return True
        return False
    except Exception as e:
        print(f"Error logging out user: {e}")
        return False


def check_password_meets_requirements(password_hash):
    """
    Check if a password meets the current requirements
    Since we can't check the actual password (it's hashed), this is just a placeholder.
    In a real application, we would need to have users update their passwords.
    
    Returns:
        bool: True if the password meets requirements, False otherwise
    """
    # This is a stub. We can't check the actual password because it's hashed.
    # We'll assume any password created before implementing the new requirements 
    # needs an update.
    return False

def login_user(username, password):
    """Verify login credentials"""
    users = _load_users()

    # Check login attempts
    can_attempt, attempt_message = check_login_attempts(username)
    if not can_attempt:
        return False, attempt_message

    # Check if username exists
    if username not in users:
        record_failed_attempt(username)
        return False, "Invalid username or password"

    # Verify password
    if not verify_password(password, users[username]["password"]):
        record_failed_attempt(username)
        return False, "Invalid username or password"

    # Reset failed attempts and generate session token
    users[username]["failed_login_attempts"] = []
    session_token, expiry = generate_session_token(username)
    users[username]["session_token"] = session_token
    users[username]["token_expiry"] = expiry
    
    # Check if password meets current requirements
    # If needs_password_update doesn't exist or is False
    if "needs_password_update" not in users[username]:
        # Set needs_password_update flag based on password requirements
        users[username]["needs_password_update"] = True
    
    _save_users(users)

    # Set session token in URL parameters - use the built-in query params API
    try:
        # For newer Streamlit versions
        current_params = st.query_params.to_dict()
        st.query_params[SESSION_TOKEN_PARAM] = session_token
        # Also include current page in URL
        if "page" in st.session_state and st.session_state.page in ["dashboard", "optimization"]:
            st.query_params[SESSION_PAGE_PARAM] = st.session_state.page
    except:
        # For compatibility with older Streamlit versions
        try:
            params = st.experimental_get_query_params()
            params[SESSION_TOKEN_PARAM] = session_token
            # Also include current page in URL
            if "page" in st.session_state and st.session_state.page in ["dashboard", "optimization"]:
                params[SESSION_PAGE_PARAM] = [st.session_state.page]
            st.experimental_set_query_params(**params)
        except:
            print("Could not set URL query parameters")

    return True, "Login successful!"


def check_login():
    """Check if user is logged in"""
    # Update last activity time
    if st.session_state.logged_in:
        st.session_state.last_activity = datetime.now()
    return st.session_state.logged_in


def get_session_time():
    """Get the current session time in minutes and seconds"""
    if "session_start_time" in st.session_state:
        elapsed = datetime.now() - st.session_state.session_start_time
        # Convert to minutes and seconds
        minutes = int(elapsed.total_seconds() // 60)
        seconds = int(elapsed.total_seconds() % 60)
        return minutes, seconds
    return 0, 0


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
                updated_portfolio = updated_users[username].get(
                    "portfolio", [])
                if len(updated_portfolio) == len(portfolio):
                    print(
                        f"Portfolio successfully saved and verified for user: {username}"
                    )
                    return True
                else:
                    print(
                        f"Error: Portfolio size mismatch after save. Expected: {len(portfolio)}, Actual: {len(updated_portfolio)}"
                    )
            else:
                print(f"Error: User '{username}' not found after saving")

        return save_result
    except Exception as e:
        print(f"Error in save_user_portfolio: {e}")
        return False


def change_password(username, old_password, new_password):
    """Change a user's password"""
    users = _load_users()

    # Check if username exists and old password is correct
    if username not in users or not verify_password(
            old_password, users[username]["password"]):
        return False
    
    # Validate new password complexity
    is_valid, _ = validate_password(new_password)
    if not is_valid:
        return False

    # Hash and save new password
    users[username]["password"] = hash_password(new_password)
    
    # Clear the needs_password_update flag since the password now meets requirements
    users[username]["needs_password_update"] = False
    
    return _save_users(users)


def get_user_portfolio(username):
    """Get the user's portfolio"""
    users = _load_users()

    # Check if username exists
    if username not in users:
        return []

    return users[username].get("portfolio", [])

# Track login attempts
login_attempts = defaultdict(list)
MAX_ATTEMPTS = 3
LOCKOUT_TIME = 1  # minutes


def check_login_attempts(username):
    """Check if user is allowed to attempt login"""
    now = datetime.now()
    # Remove old attempts
    login_attempts[username] = [
        attempt for attempt in login_attempts[username]
        if now - attempt < timedelta(minutes=LOCKOUT_TIME)
    ]

    if len(login_attempts[username]) >= MAX_ATTEMPTS:
        return False, f"Account locked for {LOCKOUT_TIME} minutes due to too many failed attempts"
    return True, ""


def record_failed_attempt(username):
    """Record a failed login attempt"""
    login_attempts[username].append(datetime.now())
