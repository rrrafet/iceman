import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color, PLOTLY_CONTINUOUS_COLORSCALE_BLUE_WHITE_RED, COLOR_PALETTE

def render_scatter_plot_with_regression(
    data_loader,
    sidebar_state,
    y_series_type: str = "portfolio_returns",
    x_series_type: str = "factor_returns",
    factor_name: str = None
) -> None:
    """SIMPLIFIED: Render scatter plot using direct schema delegation"""
    
    st.subheader("Portfolio/Active vs Factor Returns")
    
    # Get available factors using schema delegation
    factor_names = data_loader.get_factor_names()
    
    if not factor_names:
        st.info("No factor data available")
        return
    
    # Factor selector
    if factor_name is None:
        factor_name = st.selectbox(
            "Select factor for analysis",
            options=factor_names,
            key="correlation_factor_selector"
        )
    
    # Series type selector (portfolio or active)
    series_options = ["portfolio_returns", "active_returns"]
    selected_series = st.selectbox(
        "Select return series",
        options=series_options,
        format_func=lambda x: x.replace("_", " ").title(),
        key="correlation_series_selector"
    )
    
    # Get time series data using schema delegation
    component = sidebar_state.selected_node
    y_data = data_loader.get_time_series_data(selected_series, component)
    x_data = data_loader.get_time_series_data("factor_returns", factor_name)
    
    if not y_data or not x_data:
        st.info(f"Insufficient time series data for {component} vs {factor_name}: check schema time series structure")
        return
    
    # Ensure both series have same length
    min_length = min(len(y_data), len(x_data))
    if min_length < 2:
        st.info("Insufficient data points for analysis")
        return
    
    y_filtered = y_data[:min_length]
    x_filtered = x_data[:min_length]
    
    # Calculate OLS regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_filtered, y_filtered)
    
    # Create regression line points
    x_range = np.linspace(min(x_filtered), max(x_filtered), 100)
    y_regression = slope * x_range + intercept
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x_filtered,
        y=y_filtered,
        mode='markers',
        name='Returns',
        marker=dict(
            size=8,
            color=get_chart_color("active" if "active" in selected_series else "portfolio"),
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        hovertemplate=f'<b>{factor_name} Return:</b> %{{x:.3f}}<br>' +
                      f'<b>{selected_series.replace("_", " ").title()}:</b> %{{y:.3f}}<br>' +
                      '<extra></extra>'
    ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_regression,
        mode='lines',
        name=f'OLS Fit (R² = {r_value**2:.3f})',
        line=dict(
            color=COLOR_PALETTE["red"],
            width=2,
            dash='dash'
        ),
        hovertemplate=f'<b>Regression Line</b><br>' +
                      f'R² = {r_value**2:.3f}<br>' +
                      f'Beta = {slope:.3f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{selected_series.replace('_', ' ').title()} vs {factor_name} Returns",
        xaxis_title=f"{factor_name} Returns",
        yaxis_title=f"{selected_series.replace('_', ' ').title()}",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display regression statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R²", f"{r_value**2:.3f}", help="Coefficient of determination")
    with col2:
        st.metric("Beta", f"{slope:.3f}", help="Slope of regression line")
    with col3:
        st.metric("Correlation", f"{r_value:.3f}", help="Pearson correlation coefficient")
    with col4:
        p_significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        st.metric("P-value", f"{p_value:.4f}{p_significance}", help="Statistical significance")

def render_factor_correlation_heatmap(data_loader, sidebar_state) -> None:
    """SIMPLIFIED: Render heatmap using direct schema delegation"""
    
    st.subheader("Factor Correlations")
    
    # Direct schema access - single source of truth
    factor_correlations = data_loader.get_correlations("factor_correlations")
    
    if not factor_correlations:
        st.info("Factor correlation matrix not available: check schema.get_ui_matrices(TOTAL, portfolio) for correlation data")
        return
    
    # Convert to matrix format for heatmap
    factors = list(factor_correlations.keys())
    
    if not factors:
        st.info("No correlation data available")
        return
    
    # Build correlation matrix
    n_factors = len(factors)
    correlation_matrix = np.zeros((n_factors, n_factors))
    
    for i, factor1 in enumerate(factors):
        for j, factor2 in enumerate(factors):
            if factor2 in factor_correlations.get(factor1, {}):
                correlation_matrix[i, j] = factor_correlations[factor1][factor2]
            elif factor1 == factor2:
                correlation_matrix[i, j] = 1.0  # Self correlation
    
    # Create heatmap
    fig = go.Figure(go.Heatmap(
        z=correlation_matrix,
        x=factors,
        y=factors,
        colorscale=PLOTLY_CONTINUOUS_COLORSCALE_BLUE_WHITE_RED,
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(correlation_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Factor-Factor Correlations",
        height=500,
        xaxis_title="Factors",
        yaxis_title="Factors"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_portfolio_factor_correlations(data_loader, sidebar_state) -> None:
    """SIMPLIFIED: Render correlation using direct schema delegation"""
    
    st.subheader("Portfolio vs Factor Correlations")
    
    # Direct schema access - single source of truth  
    portfolio_vs_factors = data_loader.get_correlations("portfolio_vs_factors")
    
    if not portfolio_vs_factors:
        st.info("Portfolio vs factor correlations not available: check schema.get_ui_matrices(TOTAL, portfolio) for correlation data")
        return
    
    factors = list(portfolio_vs_factors.keys())
    correlations = list(portfolio_vs_factors.values())
    
    if not factors:
        st.info("No correlation data to display")
        return
    
    # Create bar chart
    fig = go.Figure(go.Bar(
        x=factors,
        y=correlations,
        marker_color=[get_chart_color("positive") if v >= 0 else get_chart_color("negative") for v in correlations],
        text=[f"{v:.3f}" for v in correlations],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Portfolio vs Factor Correlations",
        xaxis_title="Factors",
        yaxis_title="Correlation",
        height=400,
        showlegend=False
    )
    
    # Add horizontal line at zero
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    st.plotly_chart(fig, use_container_width=True)

def render_hierarchical_correlations(data_loader, sidebar_state) -> None:
    """SIMPLIFIED: Render hierarchical correlations using direct schema delegation"""
    
    st.subheader("Hierarchical Correlations")
    
    # Direct schema access - single source of truth
    hierarchical_corrs = data_loader.get_correlations("hierarchical_correlations")
    
    if not hierarchical_corrs:
        st.info("Hierarchical correlations not available: check schema.get_ui_matrices(TOTAL, portfolio) for correlation data")
        return
    
    # Display as expandable JSON
    with st.expander("View Hierarchical Correlation Data"):
        st.json(hierarchical_corrs)