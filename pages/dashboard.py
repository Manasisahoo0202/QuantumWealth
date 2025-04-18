import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.portfolio import get_portfolio_summary, get_portfolio_allocations, add_to_portfolio, remove_from_portfolio, update_portfolio_position
from utils.stock_data import validate_tickers


def show_dashboard():
    """Display the user dashboard with portfolio information"""

    st.markdown(
        f'<h1 class="main-header">Welcome, {st.session_state.username}!</h1>',
        unsafe_allow_html=True)

    # Portfolio summary, editor, and performance in different tabs
    tab1, tab2, tab3 = st.tabs(
        ["Portfolio Summary", "Portfolio Editor", "Portfolio Performance"])

    with tab1:
        show_portfolio_summary()

    with tab2:
        show_portfolio_editor()

    with tab3:
        show_portfolio_performance()


def show_portfolio_summary():
    """Display portfolio summary with current values and allocations"""

    st.markdown('<h2 class="sub-header">Portfolio Summary</h2>',
                unsafe_allow_html=True)

    try:
        # Get portfolio summary data
        summary_data = get_portfolio_summary()

        if not summary_data[0].empty:
            # Convert summary DataFrame to currency format for display
            formatted_df = summary_data[0].copy()
            for col in [
                    'purchase_price', 'current_price', 'cost_basis',
                    'current_value', 'gain_loss'
            ]:
                formatted_df[col] = formatted_df[col].map('₹{:,.2f}'.format)

            formatted_df['gain_loss_pct'] = formatted_df['gain_loss_pct'].map(
                '{:,.2f}%'.format)

            # Display the summary table
            st.dataframe(formatted_df, use_container_width=True)

            # Display total value and gain/loss
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Portfolio Value", f"₹{summary_data[1]:,.2f}")

            with col2:
                delta_color = "normal" if summary_data[2] >= 0 else "inverse"
                st.metric("Total Gain/Loss",
                          f"₹{summary_data[2]:,.2f}",
                          f"{summary_data[3]:,.2f}%",
                          delta_color=delta_color)

            # Get portfolio allocations
            allocations = get_portfolio_allocations()

            if not allocations.empty:
                with col3:
                    num_stocks = len(allocations)
                    st.metric("Number of Stocks", num_stocks)

                # Create a pie chart for allocations
                fig = px.pie(
                    allocations,
                    values='allocation',
                    names='ticker',
                    title='Portfolio Allocation',
                    template='plotly_white',
                    color_discrete_sequence=px.colors.qualitative.Pastel)

                # Customize the chart
                fig.update_traces(textposition='inside',
                                  textinfo='percent+label')
                fig.update_layout(legend=dict(orientation="h",
                                              yanchor="bottom",
                                              y=0,
                                              xanchor="center",
                                              x=0.5),
                                  margin=dict(t=30, b=0, l=0, r=0))

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "Add stocks to your portfolio to see allocation details.")
        else:
            st.info(
                "Your portfolio is empty. Add stocks using the Portfolio Editor tab."
            )

    except Exception as e:
        st.error(f"Error retrieving portfolio data: {str(e)}")
        st.info(
            "If this is your first time using the dashboard, try adding some stocks to your portfolio."
        )


def show_portfolio_editor():
    """Display interface for adding, editing, and removing portfolio holdings"""

    st.markdown('<h2 class="sub-header">Portfolio Editor</h2>',
                unsafe_allow_html=True)

    # Create tabs for add, edit, remove
    edit_tab1, edit_tab2, edit_tab3 = st.tabs(
        ["Add Stock", "Edit Position", "Remove Stock"])

    with edit_tab1:
        st.markdown("### Add a new stock to your portfolio")

        # Add instructions for Indian stocks
        st.markdown("""
        <div class="card info-card">
            <p>Enter or select an Indian NSE stock ticker (e.g., RELIANCE, TCS, HDFCBANK).</p>
        </div>
        """,
                    unsafe_allow_html=True)

        with st.form("add_stock_form"):
            # Load all NSE symbols for dropdown
            from utils.stock_data import load_nse_symbols
            nse_symbols_dict = load_nse_symbols()
            
            # Format options for the dropdown
            stock_options = []
            for symbol, company in nse_symbols_dict.items():
                stock_options.append(f"{symbol} - {company}")
            
            # Create a searchable dropdown for stocks
            selected_stock = st.selectbox(
                "Search and Select NSE Stock",
                options=stock_options,
                help="Type to search for stocks by name or symbol"
            )
            
            # Extract ticker from selection
            ticker = selected_stock.split(" - ")[0] if selected_stock else ""
            
            shares = st.number_input("Number of Shares",
                                     min_value=0.01,
                                     step=0.01)
            purchase_price = st.number_input("Purchase Price (₹)",
                                             min_value=0.01,
                                             step=0.01)

            add_submitted = st.form_submit_button("Add Stock")

            if add_submitted:
                if ticker and shares > 0 and purchase_price > 0:
                    # Validate ticker
                    valid_tickers, invalid_tickers = validate_tickers(
                        [ticker])

                    if ticker in valid_tickers:
                        # Add to portfolio
                        success = add_to_portfolio(ticker, shares,
                                                   purchase_price)

                        if success:
                            st.success(
                                f"Added {shares} shares of {ticker} to your portfolio."
                            )
                            st.rerun()
                        else:
                            st.error("Failed to add stock to portfolio.")
                    else:
                        st.error(f"Invalid ticker symbol: {ticker}")
                else:
                    st.error("Please provide valid values for all fields.")

    with edit_tab2:
        st.markdown("### Edit an existing position")

        # Get current portfolio to populate dropdown
        try:
            summary_data = get_portfolio_summary()

            if not summary_data[0].empty:
                # Get list of tickers
                tickers_list = summary_data[0]['ticker'].tolist()

                with st.form("edit_stock_form"):
                    ticker_to_edit = st.selectbox("Select Stock", tickers_list)

                    # Get current values for the selected ticker
                    current_position = summary_data[0][
                        summary_data[0]['ticker'] == ticker_to_edit]

                    current_shares = current_position['shares'].values[
                        0] if not current_position.empty else 0
                    current_price = current_position['purchase_price'].values[
                        0] if not current_position.empty else 0

                    # Input fields with current values
                    new_shares = st.number_input("Number of Shares",
                                                 min_value=0.01,
                                                 step=0.01,
                                                 value=current_shares)
                    new_price = st.number_input("Purchase Price (₹)",
                                                min_value=0.01,
                                                step=0.01,
                                                value=current_price)

                    edit_submitted = st.form_submit_button("Update Position")

                    if edit_submitted:
                        if ticker_to_edit and new_shares > 0 and new_price > 0:
                            # Update portfolio
                            success = update_portfolio_position(
                                ticker_to_edit, new_shares, new_price)

                            if success:
                                st.success(
                                    f"Updated position for {ticker_to_edit}.")
                                st.rerun()
                            else:
                                st.error("Failed to update position.")
                        else:
                            st.error(
                                "Please provide valid values for all fields.")
            else:
                st.info(
                    "Your portfolio is empty. Add stocks using the Add Stock tab."
                )

        except Exception as e:
            st.error(f"Error retrieving portfolio data: {str(e)}")

    with edit_tab3:
        st.markdown("### Remove a stock from your portfolio")

        # Get current portfolio to populate dropdown
        try:
            summary_data = get_portfolio_summary()

            if not summary_data[0].empty:
                # Get list of tickers
                tickers_list = summary_data[0]['ticker'].tolist()

                with st.form("remove_stock_form"):
                    ticker_to_remove = st.selectbox("Select Stock to Remove",
                                                    tickers_list)

                    # Add a confirmation checkbox
                    confirm = st.checkbox(
                        "I confirm I want to remove this stock from my portfolio"
                    )

                    remove_submitted = st.form_submit_button("Remove Stock")

                    if remove_submitted:
                        if ticker_to_remove and confirm:
                            # Remove from portfolio
                            success = remove_from_portfolio(ticker_to_remove)

                            if success:
                                st.success(
                                    f"Removed {ticker_to_remove} from your portfolio."
                                )
                                st.rerun()
                            else:
                                st.error(
                                    "Failed to remove stock from portfolio.")
                        else:
                            st.error(
                                "Please select a stock and confirm removal.")
            else:
                st.info(
                    "Your portfolio is empty. Add stocks using the Add Stock tab."
                )

        except Exception as e:
            st.error(f"Error retrieving portfolio data: {str(e)}")


def show_portfolio_performance():
    """Display portfolio performance charts and metrics"""

    st.markdown('<h2 class="sub-header">Portfolio Performance</h2>',
                unsafe_allow_html=True)

    try:
        # Get portfolio summary data
        summary_data = get_portfolio_summary()

        if not summary_data[0].empty:
            # Display metrics
            st.markdown("### Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                delta_color = "normal" if summary_data[3] >= 0 else "inverse"
                st.metric("Total Return",
                          f"₹{summary_data[2]:,.2f}",
                          f"{summary_data[3]:,.2f}%",
                          delta_color=delta_color)

            with col2:
                # Calculate best performing stock
                best_stock = summary_data[0].loc[summary_data[0]
                                                 ['gain_loss_pct'].idxmax()]
                st.metric("Best Performer",
                          best_stock['ticker'],
                          f"{best_stock['gain_loss_pct']:.2f}%",
                          delta_color="normal")

            with col3:
                # Calculate worst performing stock
                worst_stock = summary_data[0].loc[summary_data[0]
                                                  ['gain_loss_pct'].idxmin()]
                st.metric("Worst Performer",
                          worst_stock['ticker'],
                          f"{worst_stock['gain_loss_pct']:.2f}%",
                          delta_color="inverse")

            with col4:
                # Average return percentage
                avg_return = summary_data[0]['gain_loss_pct'].mean()
                delta_color = "normal" if avg_return >= 0 else "inverse"
                st.metric("Average Return",
                          f"{avg_return:.2f}%",
                          delta_color=delta_color)

            # Display performance chart
            st.markdown("### Individual Stock Performance")

            # Create a bar chart for individual stock performance
            fig = px.bar(
                summary_data[0],
                x='ticker',
                y='gain_loss_pct',
                color='gain_loss_pct',
                color_continuous_scale=['#EF4444', '#FFFFFF',
                                        '#10B981'],  # Red to white to green
                range_color=[
                    -max(abs(summary_data[0]['gain_loss_pct'].min()),
                         abs(summary_data[0]['gain_loss_pct'].max())),
                    max(abs(summary_data[0]['gain_loss_pct'].min()),
                        abs(summary_data[0]['gain_loss_pct'].max()))
                ],
                title='Stock Performance (%)',
                labels={
                    'ticker': 'Ticker',
                    'gain_loss_pct': 'Return (%)'
                })

            fig.update_layout(xaxis_title="Stock",
                              yaxis_title="Return (%)",
                              coloraxis_showscale=False,
                              margin=dict(l=0, r=0, t=30, b=0))

            st.plotly_chart(fig, use_container_width=True)

            # Display allocation vs performance
            st.markdown("### Allocation vs. Performance")

            # Get allocations
            allocations = get_portfolio_allocations()

            if not allocations.empty:
                # Merge with performance data
                performance_df = pd.merge(
                    allocations,
                    summary_data[0][['ticker', 'gain_loss_pct']],
                    on='ticker')

                # Create scatter plot
                fig = px.scatter(
                    performance_df,
                    x='allocation',
                    y='gain_loss_pct',
                    size='value',
                    color='gain_loss_pct',
                    color_continuous_scale=['#EF4444', '#FFFFFF', '#10B981'],
                    range_color=[
                        -max(abs(performance_df['gain_loss_pct'].min()),
                             abs(performance_df['gain_loss_pct'].max())),
                        max(abs(performance_df['gain_loss_pct'].min()),
                            abs(performance_df['gain_loss_pct'].max()))
                    ],
                    hover_name='ticker',
                    text='ticker',
                    labels={
                        'allocation': 'Portfolio Allocation (%)',
                        'gain_loss_pct': 'Return (%)',
                        'value': 'Position Value (₹)'
                    })

                fig.update_traces(textposition='top center')

                fig.update_layout(xaxis_title="Portfolio Allocation (%)",
                                  yaxis_title="Return (%)",
                                  coloraxis_showscale=False,
                                  margin=dict(l=0, r=0, t=30, b=0))

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Your portfolio is empty. Add stocks using the Portfolio Editor tab to see performance metrics."
            )

    except Exception as e:
        st.error(f"Error retrieving portfolio performance data: {str(e)}")
        st.info(
            "If this is your first time using the dashboard, try adding some stocks to your portfolio."
        )
