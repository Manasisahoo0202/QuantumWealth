import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.stock_data import get_stock_data, calculate_returns, validate_tickers, load_nse_symbols, find_high_potential_stocks
from utils.qpso import optimize_portfolio
from utils.portfolio import get_portfolio_summary, get_portfolio_allocations
import time
from datetime import datetime


def show_optimization():
    """Display the portfolio optimization page"""

    st.markdown('<h1 class="main-header">Portfolio Optimization</h1>',
                unsafe_allow_html=True)

    # Intro text
    st.markdown("""
    <div class="card">
        <p>Our portfolio optimization tool uses Quantum Particle Swarm Optimization (QPSO) algorithm to find the optimal allocation of assets
        that maximizes returns and minimizes risks based on your risk profile.</p>
    </div>
    """,
                unsafe_allow_html=True)

    # Create tabs for new portfolio and existing portfolio
    tab1, tab2 = st.tabs(["New Portfolio", "Add to Existing Portfolio"])

    with tab1:
        show_new_portfolio_optimization()

    with tab2:
        show_existing_portfolio_optimization()


def show_new_portfolio_optimization():
    """Show optimization for a new portfolio"""

    st.markdown('<h2 class="sub-header">Optimize a New Portfolio</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <p>Create an optimized portfolio by selecting Indian NSE stocks.</p>
        <p>Our QPSO algorithm will determine the ideal allocation based on your risk profile.</p>
    </div>
    """,
                unsafe_allow_html=True)

    with st.form("new_portfolio_form"):
        st.markdown("### Select NSE Stocks")
        st.markdown("""
        <div style="margin-bottom: 15px; font-size: 0.9em;">
            Search and select Indian NSE stocks for optimization
        </div>
        """,
                    unsafe_allow_html=True)

        # Load NSE symbols from CSV
        nse_symbols_dict = load_nse_symbols()

        # Create a more user-friendly format for the dropdown
        nse_options = []
        for symbol, company in nse_symbols_dict.items():
            nse_options.append(f"{symbol} - {company}")

        # Create multiselect with search capability
        st.markdown("""
        <style>
        div[data-testid="stMultiSelect"] {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #f0f2f6;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        """,
                    unsafe_allow_html=True)

        selected_nse_options = st.multiselect(
            "Search and select NSE stocks",
            options=nse_options,
            default=[],
            help=
            "Type to search for stocks by name or symbol. Select at least 2 stocks for better diversification"
        )

        # Extract just the symbol part from the selected options
        selected_stocks = [
            option.split(" - ")[0] for option in selected_nse_options
        ]

        # Show number of selected stocks
        st.markdown(f"""
        <div style="text-align: right; font-size: 0.8em; margin-top: 5px; color: #555;">
            {len(selected_stocks)} stocks selected
        </div>
        """,
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Risk Profile")

        # Risk appetite selection with better description
        risk_options = ["Conservative", "Balanced", "Aggressive"]
        risk_descriptions = {
            "Conservative":
            "Lower risk, stable returns (suitable for short-term goals)",
            "Balanced":
            "Moderate risk-return profile (suitable for medium-term goals)",
            "Aggressive":
            "Higher risk, growth oriented (suitable for long-term goals)"
        }

        risk_appetite = st.select_slider("Select your risk appetite",
                                         options=risk_options,
                                         value="Balanced")

        st.markdown(f"""
        <div style="background-color: #f7f7f7; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-size: 0.9em;">
            <strong>{risk_appetite}:</strong> {risk_descriptions[risk_appetite]}
        </div>
        """,
                    unsafe_allow_html=True)

        # Time period for historical data
        st.markdown("### Data Settings")
        period_options = {
            "3 months": "3mo",
            "6 months": "6mo",
            "1 year": "1y",
            "2 years": "2y",
            "5 years": "5y"
        }

        period_selection = st.selectbox(
            "Historical data period",
            options=list(period_options.keys()),
            index=2,
            help=
            "Longer periods provide more data but may not reflect recent trends"
        )
        period = period_options[period_selection]

        # Submit button - make it more prominent with columns
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_optimization = st.form_submit_button(
                "▶️ RUN OPTIMIZATION", use_container_width=True)

    if submit_optimization:
        if len(selected_stocks) < 2:
            st.error("Please select at least 2 stocks for optimization.")
        else:
            # Show stocks being used
            st.success(
                f"Optimizing portfolio with: {', '.join(selected_stocks)}")
            run_optimization_workflow(selected_stocks, risk_appetite, period)


def run_optimization_workflow(tickers, risk_appetite, period):
    """
    Run the optimization workflow for the given tickers

    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    risk_appetite : str
        Risk appetite (Conservative, Balanced, Aggressive)
    period : str
        Time period for historical data
    """
    # Add a header section at the top introducing the optimization process
    st.markdown("""
    <div style="background-color: #eef2ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #3b82f6;">
        <h3 style="margin-top: 0; color: #1e40af;">Portfolio Optimization Process</h3>
        <p>We're analyzing your selected stocks to create the optimal portfolio allocation based on your risk profile.</p>
    </div>
    """,
                unsafe_allow_html=True)

    # Create progress tracker
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Validate tickers
    status_text.text("Step 1/3: Validating ticker symbols...")
    progress_bar.progress(10)
    time.sleep(0.5)  # Slight delay for UX

    # Validate tickers
    valid_tickers, invalid_tickers = validate_tickers(tickers)

    progress_bar.progress(30)
    time.sleep(0.5)  # Slight delay for UX

    if invalid_tickers:
        st.warning(
            f"Some invalid ticker symbols will be ignored: {', '.join(invalid_tickers)}"
        )

    if len(valid_tickers) < 2:
        progress_bar.empty()
        status_text.empty()
        st.error(
            "At least 2 valid ticker symbols are needed for optimization.")
    else:
        st.success(
            f"Using the following valid tickers: {', '.join(valid_tickers)}")

        # Step 2: Get historical data
        status_text.text("Step 2/3: Retrieving historical stock data...")
        progress_bar.progress(50)

        # Get historical data
        data = get_stock_data(valid_tickers, period=period)

        if not data.empty:
            progress_bar.progress(70)
            time.sleep(0.5)  # Slight delay for UX

            # Calculate returns
            returns = calculate_returns(data)

            if not returns.empty:
                # Step 3: Run optimization
                status_text.text(
                    "Step 3/3: Running QPSO optimization algorithm...")
                progress_bar.progress(80)

                # Run optimization
                optimization_results = optimize_portfolio(
                    returns, risk_appetite)

                if optimization_results:
                    progress_bar.progress(100)
                    status_text.text("Optimization complete!")
                    time.sleep(0.5)  # Slight delay for UX

                    # Clear the progress elements
                    progress_bar.empty()
                    status_text.empty()

                    # Add a "success" message
                    st.markdown("""
                    <div style="background-color: #ecfdf5; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #10b981;">
                        <h3 style="margin-top: 0; color: #047857;">✅ Optimization Complete</h3>
                        <p>Your portfolio has been successfully optimized. Check out the results below.</p>
                    </div>
                    """,
                                unsafe_allow_html=True)

                    # Display results
                    show_optimization_results(optimization_results, returns)
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(
                        "Optimization failed. Please try different stocks or time period."
                    )
            else:
                progress_bar.empty()
                status_text.empty()
                st.error(
                    "Failed to calculate returns. Please try different stocks or time period."
                )
        else:
            progress_bar.empty()
            status_text.empty()
            st.error(
                "Failed to fetch historical data. Please try different stocks or time period."
            )


def show_existing_portfolio_optimization():
    """Show optimization for the user's existing portfolio"""

    st.markdown('<h2 class="sub-header">Add to Your Existing Portfolio</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <p>This tool will optimize your portfolio by suggesting additional investments while preserving your existing positions.</p>
        <p>Your current stock holdings will be maintained, and the optimizer will suggest how to allocate new investments to achieve an optimal portfolio.</p>
    </div>
    """,
                unsafe_allow_html=True)

    # Get current portfolio
    try:
        summary_data = get_portfolio_summary()

        if not summary_data[0].empty and len(summary_data[0]) >= 2:
            # Show current portfolio summary
            st.markdown("### Current Portfolio")
            st.dataframe(summary_data[0], use_container_width=True)

            with st.form("existing_portfolio_form"):
                # New investment fraction slider
                new_investment_fraction = st.slider(
                    "Percentage of current portfolio value to invest",
                    min_value=10,
                    max_value=100,
                    value=25,
                    help=
                    "What percentage of your current portfolio value do you want to add as new investment? (e.g., 25% means if your portfolio is worth $10,000, you'll invest an additional $2,500)"
                )

                # Risk appetite selection
                risk_options = ["Conservative", "Balanced", "Aggressive"]
                risk_appetite = st.select_slider(
                    "Select your risk appetite",
                    options=risk_options,
                    value="Balanced",
                    help=
                    "Conservative: Lower returns, lower risk. Aggressive: Higher potential returns, higher risk."
                )

                # Time period for historical data
                period_options = {
                    "3 months": "3mo",
                    "6 months": "6mo",
                    "1 year": "1y",
                    "2 years": "2y",
                    "5 years": "5y"
                }

                period_selection = st.selectbox(
                    "Select historical data period",
                    options=list(period_options.keys()),
                    index=2)
                period = period_options[period_selection]

                # Submit button - make it more noticeable
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    submit_optimization = st.form_submit_button(
                        "▶️ RUN OPTIMIZATION", use_container_width=True)

            if submit_optimization:
                # Get tickers from portfolio
                tickers = summary_data[0]['ticker'].tolist()

                # Use the same improved workflow as the new portfolio optimization
                # Add a header section at the top introducing the optimization process
                st.markdown("""
                <div style="background-color: #eef2ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #3b82f6;">
                    <h3 style="margin-top: 0; color: #1e40af;">Additive Portfolio Optimization</h3>
                    <p>We're analyzing your existing portfolio to determine the best way to enhance it with new investments while preserving your current holdings.</p>
                </div>
                """,
                            unsafe_allow_html=True)

                # Create progress tracker
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Get portfolio data
                status_text.text("Step 1/3: Retrieving your portfolio data...")
                progress_bar.progress(20)
                time.sleep(0.5)  # Slight delay for UX

                # Step 2: Get historical data
                status_text.text(
                    "Step 2/3: Retrieving historical stock data...")
                progress_bar.progress(50)

                # Get historical data
                data = get_stock_data(tickers, period=period)

                if not data.empty:
                    progress_bar.progress(70)
                    time.sleep(0.5)  # Slight delay for UX

                    # Calculate returns
                    returns = calculate_returns(data)

                    if not returns.empty:
                        # Step 3: Run optimization
                        status_text.text(
                            "Step 3/3: Running QPSO optimization algorithm...")
                        progress_bar.progress(80)

                        # Get current weights
                        portfolio_allocations = get_portfolio_allocations()
                        current_weights = {
                            row['ticker']: row['allocation'] / 100
                            for _, row in portfolio_allocations.iterrows()
                        } if not portfolio_allocations.empty else None

                        # Save the new investment fraction in the optimization results
                        # Run optimization with current weights, in additive mode
                        optimization_results = optimize_portfolio(
                            returns,
                            risk_appetite,
                            current_weights,
                            is_additive=True)
                        # Add the new investment fraction to results
                        optimization_results[
                            'new_investment_fraction'] = new_investment_fraction

                        if optimization_results:
                            progress_bar.progress(100)
                            status_text.text("Optimization complete!")
                            time.sleep(0.5)  # Slight delay for UX

                            # Clear the progress elements
                            progress_bar.empty()
                            status_text.empty()

                            # Add a "success" message
                            st.markdown("""
                            <div style="background-color: #ecfdf5; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #10b981;">
                                <h3 style="margin-top: 0; color: #047857;">✅ Portfolio Enhancement Complete</h3>
                                <p>Your portfolio optimization is complete. Below are the recommendations to enhance your portfolio while preserving your existing positions.</p>
                            </div>
                            """,
                                        unsafe_allow_html=True)

                            # Display results
                            show_optimization_results(optimization_results,
                                                      returns)
                        else:
                            progress_bar.empty()
                            status_text.empty()
                            st.error(
                                "Optimization failed. Please try a different time period."
                            )
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(
                            "Failed to calculate returns. Please try a different time period."
                        )
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(
                        "Failed to fetch historical data. Please try a different time period."
                    )
        else:
            # Show empty portfolio message with guidance
            st.info(
                "You need at least 2 stocks in your portfolio to run optimization."
            )
            st.markdown("""
            <div style="background-color: #f0f7ff; padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 5px solid #3b82f6;">
                <h4 style="margin-top: 0; color: #1e40af;">Getting Started</h4>
                <p>To optimize your portfolio:</p>
                <ol>
                    <li>Go to the <strong>Dashboard</strong> and add at least 2 stocks to your portfolio</li>
                    <li>Return to this page and run the optimization</li>
                    <li>Or try the "New Portfolio" tab to optimize without adding to your portfolio</li>
                </ol>
            </div>
            """,
                        unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error retrieving portfolio data: {str(e)}")


def generate_portfolio_suggestions(optimization_results, returns):
    """
    Generate AI-based suggestions for the portfolio based on optimization results

    Parameters:
    -----------
    optimization_results : dict
        Results from the optimization algorithm
    returns : pandas.DataFrame
        Returns data for the portfolio assets

    Returns:
    --------
    dict
        Dictionary of suggestions for the portfolio
    """
    suggestions = {
        "general": [],
        "high_performers": [],
        "improvement_areas": [],
        "diversification": [],
        "timing": []
    }

    try:
        # Debug print statements
        print("Generating AI investment insights...")
        print(f"Optimization results: {optimization_results.keys()}")
        print(f"Returns data shape: {returns.shape}")

        # Get the assets from the returns dataframe
        assets = returns.columns.tolist()
        # Get weights directly from optimization results
        weights = optimization_results['weights']

        print(f"Assets: {assets}")
        print(f"Weights: {weights}")

        # Calculate stats for each asset
        asset_stats = {}
        for asset in assets:
            # Check if asset is in returns columns after normalization
            col_matches = [col for col in returns.columns if asset in col]
            asset_col = col_matches[0] if col_matches else None

            if asset_col:
                annual_return = returns[asset_col].mean(
                ) * 252 * 100  # Annualized and as percentage
                volatility = returns[asset_col].std() * np.sqrt(
                    252) * 100  # Annualized and as percentage
                asset_stats[asset] = {
                    'annual_return': annual_return,
                    'volatility': volatility,
                    'sharpe':
                    annual_return / volatility if volatility > 0 else 0
                }
                print(
                    f"Stats for {asset}: annual return={annual_return:.2f}%, volatility={volatility:.2f}%, sharpe={asset_stats[asset]['sharpe']:.2f}"
                )
            else:
                print(
                    f"Warning: Asset {asset} not found in returns data columns: {returns.columns}"
                )

        # Always include these default suggestions
        suggestions["general"].append(
            "Regular review of your portfolio against your financial goals is recommended."
        )
        suggestions["diversification"].append(
            "A well-diversified portfolio helps reduce risk without sacrificing returns."
        )
        suggestions["timing"].append(
            "Consider rebalancing your portfolio quarterly to maintain the optimal allocation."
        )

        # General suggestions based on optimization results
        if optimization_results['expected_annual_return'] > 15:
            suggestions["general"].append(
                "Your portfolio shows potential for high returns, but carefully monitor market conditions."
            )
        elif optimization_results['expected_annual_return'] > 10:
            suggestions["general"].append(
                "Your portfolio has balanced growth potential while managing risk."
            )
        else:
            suggestions["general"].append(
                "Your portfolio is set for stable, conservative growth. Consider adding more growth assets if appropriate for your goals."
            )

        if optimization_results['expected_volatility'] > 25:
            suggestions["general"].append(
                "Consider reducing exposure to high-volatility assets to improve stability."
            )

        if optimization_results['sharpe_ratio'] < 0.5:
            suggestions["general"].append(
                "The risk-adjusted return (Sharpe ratio) is low. Consider rebalancing to improve return per unit of risk."
            )
        elif optimization_results['sharpe_ratio'] > 1.0:
            suggestions["general"].append(
                "Your portfolio has an excellent risk-adjusted return profile."
            )

        # High performers and areas for improvement (if we have stats)
        if asset_stats:
            # Sort by Sharpe ratio for performance analysis
            top_performers = sorted([(asset, stats['sharpe'])
                                     for asset, stats in asset_stats.items()],
                                    key=lambda x: x[1],
                                    reverse=True)[:3]
            for asset, sharpe in top_performers:
                if sharpe > 0.3:  # Lower threshold to ensure we get suggestions
                    suggestions["high_performers"].append(
                        f"{asset} is a strong performer in your portfolio with good risk-adjusted returns."
                    )
                else:
                    suggestions["high_performers"].append(
                        f"{asset} has the best risk-adjusted returns in your current selection."
                    )

            low_performers = sorted([(asset, stats['sharpe'])
                                     for asset, stats in asset_stats.items()],
                                    key=lambda x: x[1])[:3]
            for asset, sharpe in low_performers:
                if len(
                        assets
                ) > 3:  # Only show improvement suggestions if we have enough stocks
                    suggestions["improvement_areas"].append(
                        f"Consider adjusting your position in {asset} which has a lower risk-adjusted return."
                    )

        # Diversification analysis
        top_weights = sorted([(asset, weight)
                              for asset, weight in zip(assets, weights)],
                             key=lambda x: x[1],
                             reverse=True)

        if top_weights and top_weights[0][1] > 30:
            suggestions["diversification"].append(
                f"Your portfolio is heavily weighted towards {top_weights[0][0]} ({top_weights[0][1]:.1f}%). Consider diversifying to reduce concentration risk."
            )

        # Sector analysis
        financial_sector = [
            "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
            "BAJFINANCE", "BAJAJFINSV", "INDUSINDBK", "SBILIFE"
        ]
        tech_sector = ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM"]
        energy_sector = [
            "RELIANCE", "ONGC", "POWERGRID", "NTPC", "BPCL", "ADANIENT"
        ]

        for sector, sector_stocks in [("financial", financial_sector),
                                      ("technology", tech_sector),
                                      ("energy", energy_sector)]:
            sector_weight = sum(weight
                                for asset, weight in zip(assets, weights)
                                if asset in sector_stocks)
            if sector_weight > 40:
                suggestions["diversification"].append(
                    f"Your portfolio has high exposure to the {sector} sector ({sector_weight:.1f}%). Consider diversifying across more sectors."
                )

        # Additional timing suggestions
        if optimization_results['expected_volatility'] > 20:
            suggestions["timing"].append(
                "With higher portfolio volatility, more frequent rebalancing (monthly) may be beneficial."
            )

        current_date = datetime.now()
        if current_date.month in [3, 9]:  # March and September
            suggestions["timing"].append(
                "This is a good time for review and rebalancing to capture tax-loss harvesting opportunities."
            )

        print("Generated suggestions successfully")
        return suggestions

    except Exception as e:
        print(f"Error generating suggestions: {e}")
        # Return default suggestions if there's an error
        return {
            "general": [
                "Maintain a disciplined approach to your investment strategy.",
                "Regularly review your portfolio against your financial goals."
            ],
            "diversification": [
                "Consider diversifying across different asset classes and sectors to reduce risk."
            ],
            "timing": [
                "Consistent investing over time typically outperforms market timing."
            ],
            "high_performers": [
                "Focus on companies with strong fundamentals and consistent growth."
            ],
            "improvement_areas": [
                "Review underperforming positions regularly and adjust as needed."
            ]
        }


def show_optimization_results(optimization_results, returns):
    """Display the optimization results"""

    st.markdown('<h2 class="sub-header">Optimization Results</h2>',
                unsafe_allow_html=True)

    # Create columns for metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Expected Annual Return",
                  f"{optimization_results['expected_annual_return']:.2f}%")

    with col2:
        st.metric("Expected Volatility",
                  f"{optimization_results['expected_volatility']:.2f}%")

    with col3:
        st.metric("Sharpe Ratio",
                  f"{optimization_results['sharpe_ratio']:.2f}")
                  
    # AI-powered insights section removed as requested to make the UI less lengthy

    # Add a section header
    st.markdown(
        "<h3 style='color: #1e40af; margin-top: 20px;'>Understanding Your Results</h3>",
        unsafe_allow_html=True)

    # Create a card-like container with custom styling
    st.markdown(
        "<div style='background-color: #f0f7ff; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #3b82f6;'>",
        unsafe_allow_html=True)

    # Key Metrics Explained section
    st.markdown("<h4 style='color: #1e40af;'>Key Metrics Explained:</h4>",
                unsafe_allow_html=True)
    st.markdown("""
    - **Expected Annual Return:** The projected yearly return based on historical performance. Higher is better, but typically comes with higher risk.
    - **Expected Volatility:** A measure of the portfolio's risk or price fluctuation. Lower volatility means more stable returns.
    - **Sharpe Ratio:** Measures return per unit of risk. A higher Sharpe ratio indicates better risk-adjusted performance (above 1.0 is generally considered good).
    """)

    # Portfolio Allocation section
    st.markdown("<h4 style='color: #1e40af;'>Portfolio Allocation:</h4>",
                unsafe_allow_html=True)
    st.markdown("""
    The bar chart below shows the optimal allocation percentages for each asset in your portfolio. 
    These weights are calculated using our Quantum Particle Swarm Optimization (QPSO) algorithm, which finds the 
    allocation that maximizes returns while minimizing risk based on your risk profile.
    """)

    # How to Interpret section
    st.markdown("<h4 style='color: #1e40af;'>How to Interpret:</h4>",
                unsafe_allow_html=True)
    st.markdown("""
    Higher allocations to certain stocks indicate that these assets contribute more positively to your 
    portfolio's risk-return profile. Zero or low allocations suggest that including these assets would 
    increase risk without sufficient return benefit.
    """)

    # Close the container
    st.markdown("</div>", unsafe_allow_html=True)

    # Display optimal allocations
    st.markdown("### Optimal Portfolio Allocation")

    # Create DataFrame for display with proper columns
    # Get asset names from the returns dataframe
    asset_names = returns.columns.tolist()

    # Create a proper DataFrame with Asset and Weight columns
    weights_df = pd.DataFrame({
        'Asset': asset_names,
        'Weight': optimization_results['weights']
    })

    # Create a bar chart
    fig = px.bar(weights_df,
                 x='Asset',
                 y='Weight',
                 color='Weight',
                 color_continuous_scale='Blues',
                 labels={
                     'Asset': 'Ticker',
                     'Weight': 'Allocation (%)'
                 },
                 text='Weight')

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')

    fig.update_layout(xaxis_title="Stock",
                      yaxis_title="Allocation (%)",
                      coloraxis_showscale=False,
                      margin=dict(l=0, r=0, t=0, b=0))

    st.plotly_chart(fig, use_container_width=True)

    # No header needed as requested

    # Display the weights and calculate required shares
    formatted_weights = weights_df.copy()

    # Get current portfolio value
    try:
        portfolio_summary = get_portfolio_summary()[0]
        total_value = portfolio_summary['current_value'].sum()
    except Exception as e:
        print(f"Error getting portfolio value: {e}")
        total_value = 100000  # Default value if no portfolio exists

    # Get current portfolio holdings if available
    current_holdings = {}
    try:
        if not portfolio_summary.empty:
            for _, row in portfolio_summary.iterrows():
                current_holdings[row['ticker']] = row['shares']
    except Exception as e:
        print(f"Error getting current holdings: {e}")

    # Check if this is additive optimization
    is_additive = False
    if 'is_additive' in optimization_results:
        is_additive = optimization_results['is_additive']
        print(f"Using additive optimization mode: {is_additive}")

    # Calculate required investment for each asset, considering new_investment_fraction
    if is_additive and 'new_investment_fraction' in optimization_results:
        # Get the current investment value from existing portfolio
        current_investment_value = portfolio_summary['current_value'].sum()
        # Calculate new investment based on the slider value from optimization_results
        new_investment_fraction = optimization_results[
            'new_investment_fraction']
        new_investment_value = (current_investment_value *
                                new_investment_fraction / 100)
        # Total portfolio value after new investment
        future_total_value = current_investment_value + new_investment_value
        print(
            f"Current value: {current_investment_value}, New investment: {new_investment_value}, Future total: {future_total_value}"
        )
        # Calculate ONLY the new money investment (not total future value)
        formatted_weights['Target Investment'] = formatted_weights[
            'Weight'].apply(lambda x: new_investment_value * x / 100)
    else:
        # Standard calculation for new portfolio or if new_investment_fraction is not available
        formatted_weights['Target Investment'] = formatted_weights[
            'Weight'].apply(lambda x: total_value * x / 100)

    # Get current prices
    tickers = formatted_weights['Asset'].tolist()
    current_prices = get_stock_data(tickers, period='1d')
    print(
        f"MultiIndex in latest prices: {'Adj Close' in current_prices.columns and isinstance(current_prices.columns, pd.MultiIndex)}"
    )

    # Handle MultiIndex if present (from debug logs)
    if 'Adj Close' in current_prices.columns and isinstance(
            current_prices.columns, pd.MultiIndex):
        print("Using Adj Close for latest prices with MultiIndex")
        latest_prices = current_prices['Adj Close'].iloc[-1]
    else:
        latest_prices = current_prices['Adj Close'].iloc[-1]

    print(f"Latest prices type: {type(latest_prices)}")
    print(f"Latest prices index: {latest_prices.index}")

    # Convert Series to dictionary for easier access
    latest_prices_dict = {}
    for idx in latest_prices.index:
        ticker_clean = idx.replace(
            '.NS', '') if isinstance(idx, str) and '.NS' in idx else idx
        latest_prices_dict[ticker_clean] = latest_prices[idx]

    print(f"Latest prices dict: {latest_prices_dict}")

    # Calculate required shares more safely with additional logic
    def get_price(ticker):
        # Try direct match first (most common case)
        if ticker in latest_prices_dict:
            print(f"Found price for {ticker} (direct match)")
            return latest_prices_dict[ticker]

        # Try with .NS suffix
        ns_ticker = f"{ticker}.NS"
        if ns_ticker in latest_prices.index:
            print(f"Found price for {ticker} using .NS suffix")
            return latest_prices[ns_ticker]

        # Return a fallback price if not found to avoid errors
        print(
            f"Warning: Could not find price for {ticker}, using fallback price"
        )
        return 1000.0  # Default fallback price

    formatted_weights['Current Price'] = formatted_weights['Asset'].apply(
        get_price)

    # Ensure minimum share is at least 1 (can't buy partial shares)
    # Also respect current holdings in additive mode
    def calculate_shares(row):
        min_shares = 1  # Minimum is 1 share

        # Get current shares if they exist
        current_shares = current_holdings.get(row['Asset'],
                                              0) if current_holdings else 0

        # Calculate target shares based on the investment amount and current price
        target_shares = np.round(
            row['Target Investment'] / row['Current Price'], 0)

        # Always ensure we have at least the current shares (never decrease positions)
        # and at least 1 share if there's any allocation
        if row['Weight'] > 0:
            return max(current_shares, max(min_shares, target_shares))
        else:
            # If weight is 0, still maintain current shares in additive mode
            return current_shares if is_additive else 0

    formatted_weights['Shares to Hold'] = formatted_weights.apply(
        calculate_shares, axis=1)

    # Recalculate investment amounts based on actual shares to be held
    formatted_weights['Actual Investment'] = formatted_weights[
        'Shares to Hold'] * formatted_weights['Current Price']

    # Calculate the total actual investment for percentage calculation
    total_actual_investment = formatted_weights['Actual Investment'].sum()

    # Recalculate weights based on actual investment amounts
    formatted_weights['Adjusted Weight'] = formatted_weights[
        'Actual Investment'] / total_actual_investment * 100 if total_actual_investment > 0 else 0

    # Format for display
    display_weights = formatted_weights.copy()
    display_weights['Weight'] = display_weights['Weight'].map('{:.2f}%'.format)
    display_weights['Adjusted Weight'] = display_weights[
        'Adjusted Weight'].map('{:.2f}%'.format)
    display_weights['Target Investment'] = display_weights[
        'Target Investment'].map('₹{:,.0f}'.format)
    display_weights['Actual Investment'] = display_weights[
        'Actual Investment'].map('₹{:,.0f}'.format)
    display_weights['Current Price'] = display_weights['Current Price'].map(
        '₹{:,.2f}'.format)

    # Create styled table for stock quantities
    st.markdown("""
    <style>
    .stock-table {
        font-size: 1.1em;
        margin: 10px 0;
    }
    .stock-table th {
        text-align: left;
        padding: 12px;
        background-color: #f1f5f9;
        color: #1e40af;
    }
    .stock-table td {
        padding: 12px;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """,
                unsafe_allow_html=True)

    # Use Streamlit's native table to display the results
    st.subheader("Recommended Portfolio Allocation")

    # Create a cleaner display table
    display_df = display_weights.copy()
    display_df = display_df.rename(
        columns={
            'Asset': 'Stock',
            'Weight': 'Target Weight',
            'Adjusted Weight': 'Actual Weight',
            'Current Price': 'Current Price',
            'Shares to Hold': 'Required Quantity',
            'Actual Investment': 'Investment Amount'
        })

    # Select and order the columns for display
    display_df = display_df[[
        'Stock', 'Target Weight', 'Actual Weight', 'Current Price',
        'Required Quantity', 'Investment Amount'
    ]]

    # Round Required Quantity to whole numbers
    display_df['Required Quantity'] = display_df['Required Quantity'].astype(
        int)

    # Display the table
    st.dataframe(display_df, use_container_width=True)

    # Calculate and display total investment value - use actual investment
    total_investment = formatted_weights['Actual Investment'].sum()
    st.markdown(f"""
    <div style="background-color: #ecfdf5; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #10b981; text-align: center; font-size: 1.2em;">
        <strong>Total Investment Value:</strong> ₹{total_investment:,.2f}
    </div>
    """,
                unsafe_allow_html=True)

    # No explanation text as requested
    
    # High Potential Stocks section
    st.markdown("<hr style='margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style="color: #1e40af; margin-bottom: 20px;">High-Potential Stock Recommendations</h2>
    <div style="background-color: #f0f7ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #3b82f6;">
        <p>Discover high-potential NSE stocks that aren't in your current portfolio. These recommendations are based on historical performance, risk-adjusted returns, and recent momentum.</p>
    </div>
    """, unsafe_allow_html=True)
    
    print("Starting high-potential stock recommendations section")
    
    # Show a spinner while loading recommendations
    with st.spinner("Finding high-potential stocks..."):
        # Get current tickers from the optimization
        print(f"Returns object type: {type(returns)}")
        print(f"Returns columns: {returns.columns if hasattr(returns, 'columns') else 'No columns attribute'}")
        current_tickers = [col.split('.')[0] if '.' in col else col for col in returns.columns]
        print(f"Extracted current tickers: {current_tickers}")
        
        # Find high potential stocks not in the current portfolio
        high_potential_results = find_high_potential_stocks(
            current_tickers=current_tickers,
            period="1y",  # Use 1 year of data for consistent analysis
            top_n=5  # Show top 5 recommendations
        )
        
        if "high_return_stocks" in high_potential_results and high_potential_results["high_return_stocks"]:
            # Create columns for displaying the stocks
            reco_stocks = high_potential_results["high_return_stocks"]
            
            # Create multiple columns to display the stocks
            cols = st.columns(min(len(reco_stocks), 3))
            
            for i, stock in enumerate(reco_stocks[:min(len(reco_stocks), 3)]):
                with cols[i]:
                    # Create a card-like display for each stock
                    st.markdown(f"""
                    <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; height: 100%;">
                        <h3 style="color: #1e40af; margin-top: 0;">{stock['ticker']}</h3>
                        <p style="color: #4b5563; font-size: 0.9em; margin-bottom: 15px;">{stock['company_name']}</p>
                        <div style="margin: 12px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: #4b5563;">Annual Return:</span>
                                <span style="color: {'#10b981' if stock['annual_return'] > 0 else '#ef4444'}; font-weight: bold;">{stock['annual_return']:.2f}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: #4b5563;">Volatility:</span>
                                <span style="color: #4b5563;">{stock['volatility']:.2f}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: #4b5563;">Sharpe Ratio:</span>
                                <span style="color: #4b5563;">{stock['sharpe_ratio']:.2f}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: #4b5563;">Momentum:</span>
                                <span style="color: {'#10b981' if stock['momentum'] > 1 else '#4b5563'};">{stock['momentum']:.2f}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if len(reco_stocks) > 3:
                # Create another row for additional stocks
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                cols2 = st.columns(min(len(reco_stocks) - 3, 3))
                
                for i, stock in enumerate(reco_stocks[3:min(len(reco_stocks), 6)]):
                    with cols2[i]:
                        # Create a card-like display for each stock
                        st.markdown(f"""
                        <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; height: 100%;">
                            <h3 style="color: #1e40af; margin-top: 0;">{stock['ticker']}</h3>
                            <p style="color: #4b5563; font-size: 0.9em; margin-bottom: 15px;">{stock['company_name']}</p>
                            <div style="margin: 12px 0;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                    <span style="color: #4b5563;">Annual Return:</span>
                                    <span style="color: {'#10b981' if stock['annual_return'] > 0 else '#ef4444'}; font-weight: bold;">{stock['annual_return']:.2f}%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                    <span style="color: #4b5563;">Volatility:</span>
                                    <span style="color: #4b5563;">{stock['volatility']:.2f}%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                    <span style="color: #4b5563;">Sharpe Ratio:</span>
                                    <span style="color: #4b5563;">{stock['sharpe_ratio']:.2f}</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                    <span style="color: #4b5563;">Momentum:</span>
                                    <span style="color: {'#10b981' if stock['momentum'] > 1 else '#4b5563'};">{stock['momentum']:.2f}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Add explanation for metrics
            st.markdown("""
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin: 20px 0; font-size: 0.9em;">
                <h4 style="color: #1e40af; margin-top: 0;">Understanding the Metrics</h4>
                <ul style="margin-top: 10px; padding-left: 20px;">
                    <li><strong>Annual Return:</strong> The historical yearly return of the stock based on past performance.</li>
                    <li><strong>Volatility:</strong> A measure of the stock's price fluctuation. Lower values indicate more stable performance.</li>
                    <li><strong>Sharpe Ratio:</strong> A measure of risk-adjusted return. Higher values indicate better returns for the risk taken.</li>
                    <li><strong>Momentum:</strong> Indicates recent performance trend. Values above 1.0 suggest improving performance.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            # No high potential stocks found or error
            error_message = high_potential_results.get("error", "No high-potential stocks found matching your criteria.")
            st.info(f"Unable to generate stock recommendations: {error_message}")
            st.markdown("""
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <p>Try the following:</p>
                <ul>
                    <li>Ensure you have a stable internet connection to fetch market data</li>
                    <li>Try again later when market data becomes available</li>
                    <li>Consider adjusting your existing portfolio with more diverse stocks</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


