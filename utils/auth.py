import streamlit as st
import json
import os
from passlib.hash import bcrypt
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from datetime import datetime, timedelta
# Create a directory for storing user data
os.makedirs("data", exist_ok=True)
# Use absolute path for reliability
USERS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
    "users.json")
print(f"Users file path: {USERS_FILE}")


def initialize_session_state():
    """Initialize session state variables if they don't exist"""
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
        "needs_password_update": False,
        "failed_login_attempts": []
    }

    # Check existing accounts for password complexity requirements
    for user in users:
        if "needs_password_update" not in users[user]:
            users[user]["needs_password_update"] = True
        if "failed_login_attempts" not in users[user]:
            users[user]["failed_login_attempts"] = []

    # Save users
    if _save_users(users):
        return True, "Registration successful"
    else:
        return False, "Error saving user data"


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

    # Reset failed attempts on successful login
    users[username]["failed_login_attempts"] = []
    _save_users(users)

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

    # Hash and save new password
    users[username]["password"] = hash_password(new_password)
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