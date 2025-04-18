import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from streamlit.web.server.websocket_headers import _get_websocket_headers
import os
from datetime import datetime, timedelta
import json
import numpy as np
import sys

# Import utils
from utils.auth import initialize_session_state, register_user, login_user, check_login
from utils.portfolio import (
    add_to_portfolio, 
    remove_from_portfolio, 
    update_portfolio_position, 
    get_portfolio_summary, 
    get_portfolio_allocations
)
from utils.stock_data import get_stock_data, calculate_returns, validate_tickers
from utils.qpso import optimize_portfolio
from pages.welcome import show_welcome
from pages.dashboard import show_dashboard
from pages.optimization import show_optimization

# Set page configuration
st.set_page_config(
    page_title="QuantumWealth - NSE Portfolio Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None  # Remove menu items completely
)

# Initialize session state
initialize_session_state()

# Get the current URL path
def get_current_path():
    try:
        ws_headers = _get_websocket_headers()
        uri = ws_headers.get("URI", "")
        # Extract path from URI
        path = uri.split("?")[0] if "?" in uri else uri
        return path.strip("/")
    except:
        return ""

# Define CSS
st.markdown("""
<style>
    /* Hide sidebar, hamburger menu, and menu button */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    button[kind="headerNoPadding"] {
        display: none !important;
    }
    
    .stDeployButton {
        display: none !important;
    }
    
    /* Adjust main content */
    .main {
        padding-top: 0;
    }
    
    /* Header styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.8rem;
        color: #1E3A8A;
    }
    
    /* App header styling */
    .app-header {
        background-color: white;
        padding: 1rem 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
        z-index: 999;
        width: 100%;
    }
    
    .header-left {
        display: flex;
        align-items: center;
    }
    
    .header-right {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .nav-btn {
        background-color: #f0f7ff;
        border: 1px solid #e2e8f0;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
        color: #1E3A8A;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .nav-btn:hover {
        background-color: #e2e8f0;
    }
    
    .nav-btn.active {
        background-color: #1E3A8A;
        color: white;
    }
    
    .logout-btn {
        background-color: #fee2e2;
        border: 1px solid #fecaca;
        color: #b91c1c;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        cursor: pointer;
    }
    
    .logout-btn:hover {
        background-color: #fecaca;
    }
    
    .logo-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-left: 0.5rem;
    }
    
    .username-display {
        background-color: #f0f9ff;
        border-radius: 1rem;
        padding: 0.3rem 1rem;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        margin-right: 1rem;
    }
    
    /* Card component */
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        font-size: 0.8rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Create a native Streamlit header with logo, navigation, and user info
st.markdown("""
<style>
    /* Custom header styling */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        margin-bottom: 20px;
        background-color: white;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .header-left {
        display: flex;
        align-items: center;
    }
    
    .header-right {
        display: flex;
        align-items: center;
    }
    
    .username-badge {
        margin-right: 15px;
        background-color: #f0f9ff;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 14px;
    }
    
    .logo-img {
        width: 40px;
        height: 40px;
        margin-right: 10px;
    }
    
    .logo-text {
        font-size: 20px;
        font-weight: bold;
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Create header with columns
header_container = st.container()
with header_container:
    col1, col2 = st.columns([2, 3])
    
    # Logo and app name on the left
    with col1:
        logo_col, text_col = st.columns([1, 4])
        with logo_col:
            st.image("assets/quantum_wealth_logo.svg", width=40)
        with text_col:
            st.markdown('<span class="logo-text">QuantumWealth</span>', unsafe_allow_html=True)
    
    # Navigation on the right
    with col2:
        if st.session_state.logged_in:
            # Create right-aligned columns for username and buttons
            rcol1, rcol2, rcol3, rcol4 = st.columns([2, 1, 1, 1])
            
            with rcol1:
                st.markdown(f"""
                <div class="username-badge">
                    👤 {st.session_state.username}
                </div>
                """, unsafe_allow_html=True)
                
            with rcol2:
                dashboard_active = st.session_state.page == "dashboard"
                if st.button("📊 Dashboard", 
                           type="primary" if dashboard_active else "secondary",
                           use_container_width=True):
                    st.session_state.page = "dashboard"
                    st.rerun()
                    
            with rcol3:
                optimize_active = st.session_state.page == "optimization"
                if st.button("⚡ Optimize", 
                           type="primary" if optimize_active else "secondary",
                           use_container_width=True):
                    st.session_state.page = "optimization"
                    st.rerun()
            
            with rcol4:
                if st.button("🚪 Logout", type="secondary", use_container_width=True):
                    st.session_state.logged_in = False
                    st.session_state.username = ""
                    st.session_state.page = "welcome"
                    st.rerun()
        else:
            # For non-logged-in users, show the app description
            st.markdown("""
            <div style="text-align: right; padding-right: 20px; font-weight: bold; color: #1E3A8A;">
                NSE Portfolio Optimizer
            </div>
            """, unsafe_allow_html=True)

# Add a separator
st.markdown("<hr style='margin: 0; padding: 0; height: 1px; background-color: #e2e8f0; border: none;'>", 
           unsafe_allow_html=True)
    
# Main content
if not st.session_state.logged_in:
    show_welcome()
elif st.session_state.page == "dashboard":
    show_dashboard()
elif st.session_state.page == "optimization":
    show_optimization()
else:
    show_welcome()

