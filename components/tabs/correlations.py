import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from components.charts.correlation_charts import (
    render_scatter_plot_with_regression,
    render_factor_correlation_heatmap, 
    render_portfolio_factor_correlations,
    render_hierarchical_correlations
)

def render_correlations_tab(data_loader, sidebar_state):
    """Render Tab 10 - Correlations with new scatter plot"""
    
    st.header("Correlations - Factor & Portfolio Analysis")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # NEW FEATURE: Scatter plot of timeseries (y-axis) portfolio/active vs factor returns (x-axis) with OLS regression
    render_scatter_plot_with_regression(data_loader, sidebar_state)
    
    st.divider()
    
    # Two columns for different correlation analyses
    col1, col2 = st.columns(2)
    
    with col1:
        # Factor-factor correlation heatmap
        render_factor_correlation_heatmap(data_loader, sidebar_state)
    
    with col2:
        # Portfolio vs factors correlation bars
        render_portfolio_factor_correlations(data_loader, sidebar_state)
    
    st.divider()
    
    # Hierarchical correlations (if available)
    render_hierarchical_correlations(data_loader, sidebar_state)
    
    st.divider()
    
    # Explanation section
    st.subheader("Understanding Correlations")
    
    st.markdown("""
    **Interpretation Guide:**
    
    • **Scatter Plot with Regression**: Shows the relationship between portfolio/active returns and individual factor returns over time. 
      The OLS regression line helps identify systematic factor exposure.
    
    • **R² Value**: Measures how much of the portfolio's variance is explained by the factor (0-1 scale)
    
    • **Beta**: The sensitivity of portfolio returns to factor returns. Beta > 1 indicates amplified exposure.
    
    • **Factor Correlations**: Shows how factors move together. High correlations may indicate factor concentration risk.
    
    • **Portfolio vs Factor Correlations**: Identifies which factors drive portfolio performance.
    """)
    
    # Data availability status
    st.subheader("Data Status")
    
    # Check what correlation data is available
    factor_corrs = data_loader.get_correlations("factor_correlations")
    portfolio_factor_corrs = data_loader.get_correlations("portfolio_vs_factors")
    hierarchical_corrs = data_loader.get_correlations("hierarchical_correlations")
    
    status_info = []
    
    if factor_corrs:
        status_info.append("Factor-factor correlations available")
    else:
        status_info.append("Factor-factor correlations will populate when computed")
    
    if portfolio_factor_corrs:
        status_info.append("Portfolio-factor correlations available")
    else:
        status_info.append("Portfolio-factor correlations will populate when computed")
    
    if hierarchical_corrs:
        status_info.append("Hierarchical correlations available")
    else:
        status_info.append("Hierarchical correlations will populate when computed")
    
    # Time series data availability
    factor_names = data_loader.get_factor_names()
    time_series = data_loader.data.get('time_series', {})
    factor_returns = time_series.get('factor_returns', {})
    
    available_factor_series = len([f for f in factor_names if f in factor_returns])
    status_info.append(f"Factor return series: {available_factor_series}/{len(factor_names)} available")
    
    # Portfolio/active return series
    component = sidebar_state.selected_node
    portfolio_data = data_loader.get_time_series_data('portfolio_returns', component)
    active_data = data_loader.get_time_series_data('active_returns', component)
    
    if portfolio_data:
        status_info.append(f"Portfolio returns: {len(portfolio_data)} periods available")
    else:
        status_info.append("Portfolio returns not available for selected component")
    
    if active_data:
        status_info.append(f"Active returns: {len(active_data)} periods available")
    else:
        status_info.append("Active returns not available for selected component")
    
    # Display status
    for status in status_info:
        st.markdown(f"• {status}")
    
    # Date range info
    if sidebar_state.date_range != (0, 59):  # If not default full range
        start_idx, end_idx = sidebar_state.date_range
        st.info(f"Analysis filtered to periods {start_idx} to {end_idx} ({end_idx - start_idx + 1} periods)")
    
    # Factor filter info
    if sidebar_state.selected_factors:
        st.info(f"Analysis limited to {len(sidebar_state.selected_factors)} selected factors: {', '.join(sidebar_state.selected_factors)}")
    else:
        st.info("All factors included in correlation analysis")