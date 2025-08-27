import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color

def render_stats_distributions_tab(data_loader, sidebar_state):
    """Render Tab 9 - Stats & distributions"""
    
    st.header("Statistics & Distributions")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # Snapshot stats for current node
    render_snapshot_statistics(data_loader, sidebar_state)
    
    st.divider()
    
    # Distributions section
    render_distributions_analysis(data_loader, sidebar_state)

def render_snapshot_statistics(data_loader, sidebar_state):
    """Render snapshot statistics for current node"""
    
    st.subheader("Component Statistics")
    
    current_node = sidebar_state.selected_node
    
    # Get time series data for statistics calculation
    portfolio_data = data_loader.get_time_series_data('portfolio_returns', current_node)
    benchmark_data = data_loader.get_time_series_data('benchmark_returns', current_node)
    active_data = data_loader.get_time_series_data('active_returns', current_node)
    
    # Apply date range filter if needed
    start_idx, end_idx = sidebar_state.date_range
    if portfolio_data and (start_idx > 0 or end_idx < len(portfolio_data) - 1):
        portfolio_data = portfolio_data[start_idx:end_idx + 1]
    if benchmark_data and (start_idx > 0 or end_idx < len(benchmark_data) - 1):
        benchmark_data = benchmark_data[start_idx:end_idx + 1]
    if active_data and (start_idx > 0 or end_idx < len(active_data) - 1):
        active_data = active_data[start_idx:end_idx + 1]
    
    # Three columns for different return series
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_series_statistics("Portfolio", portfolio_data, sidebar_state)
    
    with col2:
        render_series_statistics("Benchmark", benchmark_data, sidebar_state)
    
    with col3:
        render_series_statistics("Active", active_data, sidebar_state)

def render_series_statistics(series_name, data, sidebar_state):
    """Render statistics for a single time series"""
    
    st.markdown(f"**{series_name} Statistics**")
    
    if not data or len(data) < 2:
        st.info(f"No {series_name.lower()} data available")
        return
    
    # Calculate statistics
    stats = calculate_time_series_statistics(data)
    
    # Display key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mean", f"{stats['mean']:.4f}")
        st.metric("Std Dev", f"{stats['std']:.4f}")
        st.metric("Min", f"{stats['min']:.4f}")
        st.metric("Max", f"{stats['max']:.4f}")
    
    with col2:
        st.metric("Skewness", f"{stats['skew']:.3f}")
        st.metric("Kurtosis", f"{stats['kurtosis']:.3f}")
        
        # Annualized metrics (assuming daily data, 252 trading days)
        annualization_factor = 252 if len(data) > 252 else len(data)
        ann_return = stats['mean'] * annualization_factor
        ann_vol = stats['std'] * np.sqrt(annualization_factor)
        
        st.metric("Ann. Return", f"{ann_return:.2%}")
        st.metric("Ann. Vol", f"{ann_vol:.2%}")
    
    # Sharpe ratio (using mean return as proxy for excess return)
    if stats['std'] > 0:
        sharpe = stats['mean'] / stats['std'] * np.sqrt(annualization_factor)
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")

def calculate_time_series_statistics(data):
    """Calculate comprehensive statistics for a time series"""
    
    if not data:
        return {}
    
    data_array = np.array(data)
    
    # Basic statistics
    mean = np.mean(data_array)
    std = np.std(data_array, ddof=1)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    # Higher moments
    skew = calculate_skewness(data_array, mean, std)
    kurtosis = calculate_kurtosis(data_array, mean, std)
    
    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'skew': skew,
        'kurtosis': kurtosis
    }

def calculate_skewness(data, mean, std):
    """Calculate skewness"""
    if std == 0:
        return 0
    
    n = len(data)
    skew = np.sum(((data - mean) / std) ** 3) / n
    return skew

def calculate_kurtosis(data, mean, std):
    """Calculate excess kurtosis"""
    if std == 0:
        return 0
    
    n = len(data)
    kurt = np.sum(((data - mean) / std) ** 4) / n - 3  # Excess kurtosis
    return kurt

def render_distributions_analysis(data_loader, sidebar_state):
    """Render distribution analysis section"""
    
    st.subheader("Distribution Analysis")
    
    # Two columns for different distribution views
    col1, col2 = st.columns(2)
    
    with col1:
        render_return_distributions(data_loader, sidebar_state)
    
    with col2:
        render_factor_return_distributions(data_loader, sidebar_state)

def render_return_distributions(data_loader, sidebar_state):
    """Render histogram and KDE for portfolio vs benchmark returns"""
    
    st.markdown("**Return Distributions**")
    
    current_node = sidebar_state.selected_node
    
    # Get return data
    portfolio_data = data_loader.get_time_series_data('portfolio_returns', current_node)
    benchmark_data = data_loader.get_time_series_data('benchmark_returns', current_node)
    
    # Apply date range filter
    start_idx, end_idx = sidebar_state.date_range
    if portfolio_data and (start_idx > 0 or end_idx < len(portfolio_data) - 1):
        portfolio_data = portfolio_data[start_idx:end_idx + 1]
    if benchmark_data and (start_idx > 0 or end_idx < len(benchmark_data) - 1):
        benchmark_data = benchmark_data[start_idx:end_idx + 1]
    
    if not portfolio_data:
        st.info("No return data available for distribution analysis")
        return
    
    # Create histogram
    fig = go.Figure()
    
    # Portfolio distribution
    fig.add_trace(go.Histogram(
        x=portfolio_data,
        name='Portfolio',
        nbinsx=30,
        opacity=0.7,
        marker_color=get_chart_color("portfolio"),
        hovertemplate='Return: %{x:.4f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Benchmark distribution (if available)
    if benchmark_data:
        fig.add_trace(go.Histogram(
            x=benchmark_data,
            name='Benchmark',
            nbinsx=30,
            opacity=0.7,
            marker_color=get_chart_color("benchmark"),
            hovertemplate='Return: %{x:.4f}<br>Count: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Return Distribution - {current_node}",
        xaxis_title="Return",
        yaxis_title="Frequency",
        height=400,
        barmode='overlay',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution comparison metrics
    if benchmark_data:
        render_distribution_comparison(portfolio_data, benchmark_data)

def render_distribution_comparison(portfolio_data, benchmark_data):
    """Render comparison metrics between portfolio and benchmark distributions"""
    
    st.markdown("**Distribution Comparison**")
    
    port_stats = calculate_time_series_statistics(portfolio_data)
    bench_stats = calculate_time_series_statistics(benchmark_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mean_diff = port_stats['mean'] - bench_stats['mean']
        st.metric("Mean Difference", f"{mean_diff:.4f}", 
                 delta=f"{mean_diff:.4f}")
    
    with col2:
        vol_diff = port_stats['std'] - bench_stats['std']
        st.metric("Volatility Difference", f"{vol_diff:.4f}",
                 delta=f"{vol_diff:.4f}")
    
    with col3:
        # Tracking error (std of active returns)
        active_returns = [p - b for p, b in zip(portfolio_data, benchmark_data)]
        tracking_error = np.std(active_returns, ddof=1) if len(active_returns) > 1 else 0
        st.metric("Tracking Error", f"{tracking_error:.4f}")

def render_factor_return_distributions(data_loader, sidebar_state):
    """Render box/violin plots for factor returns"""
    
    st.markdown("**Factor Return Distributions**")
    
    # Get factor names
    factor_names = data_loader.get_factor_names()
    
    # Filter by selected factors if any
    if sidebar_state.selected_factors:
        display_factors = [f for f in factor_names if f in sidebar_state.selected_factors]
    else:
        display_factors = factor_names[:8]  # Limit to first 8 for display
    
    if not display_factors:
        st.info("No factors selected for distribution analysis")
        return
    
    # Factor selection
    selected_factors = st.multiselect(
        "Select factors for distribution analysis",
        options=factor_names,
        default=display_factors[:5],  # Default to first 5
        max_selections=8,
        key="factor_distribution_selector"
    )
    
    if not selected_factors:
        st.info("Select factors to view distributions")
        return
    
    # Create box plots
    fig = go.Figure()
    
    for factor in selected_factors:
        factor_data = data_loader.get_time_series_data('factor_returns', factor)
        
        if factor_data:
            # Apply date range filter
            start_idx, end_idx = sidebar_state.date_range
            if start_idx > 0 or end_idx < len(factor_data) - 1:
                factor_data = factor_data[start_idx:end_idx + 1]
            
            fig.add_trace(go.Box(
                y=factor_data,
                name=factor,
                boxpoints='outliers',
                hovertemplate=f'{factor}<br>Value: %{{y:.4f}}<extra></extra>'
            ))
    
    if fig.data:
        fig.update_layout(
            title="Factor Return Distributions",
            xaxis_title="Factors",
            yaxis_title="Returns",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Factor distribution statistics
        render_factor_distribution_stats(selected_factors, data_loader, sidebar_state)
    else:
        st.info("No factor return data available")

def render_factor_distribution_stats(factors, data_loader, sidebar_state):
    """Render summary statistics for factor distributions"""
    
    st.markdown("**Factor Distribution Summary**")
    
    # Create summary table
    import pandas as pd
    
    summary_data = []
    
    for factor in factors:
        factor_data = data_loader.get_time_series_data('factor_returns', factor)
        
        if factor_data:
            # Apply date range filter
            start_idx, end_idx = sidebar_state.date_range
            if start_idx > 0 or end_idx < len(factor_data) - 1:
                factor_data = factor_data[start_idx:end_idx + 1]
            
            stats = calculate_time_series_statistics(factor_data)
            
            summary_data.append({
                'Factor': factor,
                'Mean': f"{stats['mean']:.4f}",
                'Std Dev': f"{stats['std']:.4f}",
                'Skew': f"{stats['skew']:.3f}",
                'Kurtosis': f"{stats['kurtosis']:.3f}",
                'Min': f"{stats['min']:.4f}",
                'Max': f"{stats['max']:.4f}"
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Interpretation guide
        with st.expander("Distribution Interpretation Guide"):
            st.markdown("""
            **Statistical Interpretation:**
            
            - **Mean**: Average return over the period
            - **Std Dev**: Volatility/risk measure
            - **Skewness**: Distribution asymmetry
              - Positive: More extreme positive returns
              - Negative: More extreme negative returns
              - Close to 0: Symmetric distribution
            - **Kurtosis**: Tail thickness (excess over normal distribution)
              - Positive: Fatter tails (more extreme events)
              - Negative: Thinner tails
              - Close to 0: Normal-like distribution
            """)
    else:
        st.info("No factor data available for statistics")