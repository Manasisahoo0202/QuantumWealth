import streamlit as st
import pandas as pd
import numpy as np
from utils.auth import register_user, login_user

def show_welcome():
    """Display the welcome page with login and registration forms"""
    
    # Main header
    st.markdown('<h1 class="main-header">Welcome to QuantumWealth</h1>', unsafe_allow_html=True)
    
    # Intro
    st.markdown("""
    <div class="card">
        <p>QuantumWealth is an advanced portfolio optimization platform that uses quantum-inspired algorithms to help you maximize returns and minimize risks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for login and registration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="sub-header">Login</h2>', unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login")
            
            if submit_login:
                if username and password:
                    if login_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.page = "dashboard"
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
    
    with col2:
        st.markdown('<h2 class="sub-header">Register</h2>', unsafe_allow_html=True)
        
        # Registration form
        with st.form("registration_form"):
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_registration = st.form_submit_button("Register")
            
            if submit_registration:
                if new_username and new_password and confirm_password:
                    if len(new_username) < 3:
                        st.error("Username must be at least 3 characters")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        success = register_user(new_username, new_password)
                        if success:
                            st.success("Registration successful! You can now log in.")
                            # Auto-login after registration
                            st.session_state.logged_in = True
                            st.session_state.username = new_username
                            st.session_state.page = "dashboard"
                            st.rerun()
                        else:
                            st.error("Username already exists")
                else:
                    st.error("Please fill in all fields")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>QuantumWealth uses advanced algorithms to analyze historical market data and provide investment recommendations. 
        Past performance is not indicative of future results. Investment involves risk.</p>
        <p style="margin-top: 20px;">© 2025 QuantumWealth. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)