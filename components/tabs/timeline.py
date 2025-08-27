import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color

def render_timeline_tab(data_loader, sidebar_state):
    """Render Tab 4 - Timeline (returns)"""
    
    st.header("Timeline - Returns Analysis")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # Date selector (mirrors sidebar)
    render_date_range_info(sidebar_state)
    
    st.divider()
    
    # Main time series chart
    render_main_time_series(data_loader, sidebar_state)
    
    st.divider()
    
    # Multi-panel picker for child components
    render_multi_component_picker(data_loader, sidebar_state)
    
    st.divider()
    
    # Rolling statistics section
    render_rolling_statistics(data_loader, sidebar_state)

def render_date_range_info(sidebar_state):
    """Display current date range selection"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_idx, end_idx = sidebar_state.date_range
        st.metric("Start Period", start_idx)
    
    with col2:
        st.metric("End Period", end_idx)
    
    with col3:
        total_periods = end_idx - start_idx + 1
        st.metric("Total Periods", total_periods)
    
    if sidebar_state.date_range != (0, 59):  # If not default full range
        st.info(f"Showing periods {start_idx} to {end_idx} ({total_periods} periods)")
    else:
        st.info("Showing full time series")

def render_main_time_series(data_loader, sidebar_state):
    """Render main time series chart with portfolio, benchmark, and active returns"""
    
    st.subheader("Returns Time Series")
    
    current_node = sidebar_state.selected_node
    
    # Get time series data
    portfolio_data = data_loader.get_time_series_data('portfolio_returns', current_node)
    benchmark_data = data_loader.get_time_series_data('benchmark_returns', current_node)
    active_data = data_loader.get_time_series_data('active_returns', current_node)
    
    if not portfolio_data:
        st.info(f"No time series data available for {current_node}")
        return
    
    # Apply date range filter
    start_idx, end_idx = sidebar_state.date_range
    if start_idx > 0 or end_idx < len(portfolio_data) - 1:
        portfolio_data = portfolio_data[start_idx:end_idx + 1]
        if benchmark_data:
            benchmark_data = benchmark_data[start_idx:end_idx + 1]
        if active_data:
            active_data = active_data[start_idx:end_idx + 1]
    
    # Create date index
    periods = list(range(len(portfolio_data)))
    
    # Create figure
    fig = go.Figure()
    
    # Add portfolio returns
    fig.add_trace(go.Scatter(
        x=periods,
        y=portfolio_data,
        mode='lines',
        name='Portfolio',
        line=dict(color=get_chart_color("portfolio"), width=2),
        hovertemplate='Period: %{x}<br>Portfolio Return: %{y:.4f}<extra></extra>'
    ))
    
    # Add benchmark returns if available
    if benchmark_data:
        fig.add_trace(go.Scatter(
            x=periods,
            y=benchmark_data,
            mode='lines',
            name='Benchmark',
            line=dict(color=get_chart_color("benchmark"), width=2, dash='dash'),
            hovertemplate='Period: %{x}<br>Benchmark Return: %{y:.4f}<extra></extra>'
        ))
    
    # Add active returns if available and if toggle is enabled
    show_active = st.toggle("Show Active Returns", value=True, key="show_active_returns")
    
    if show_active and active_data:
        fig.add_trace(go.Scatter(
            x=periods,
            y=active_data,
            mode='lines',
            name='Active',
            line=dict(color=get_chart_color("active"), width=2),
            hovertemplate='Period: %{x}<br>Active Return: %{y:.4f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Returns Time Series - {current_node}",
        xaxis_title="Period",
        yaxis_title="Return",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display summary statistics
    render_series_summary_stats(portfolio_data, benchmark_data, active_data, show_active)

def render_series_summary_stats(portfolio_data, benchmark_data, active_data, show_active):
    """Display summary statistics for the time series"""
    
    st.markdown("**Series Summary Statistics**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if portfolio_data:
            port_mean = sum(portfolio_data) / len(portfolio_data)
            port_std = (sum([(x - port_mean)**2 for x in portfolio_data]) / len(portfolio_data))**0.5
            
            st.markdown("**Portfolio**")
            st.metric("Mean Return", f"{port_mean:.4f}")
            st.metric("Volatility", f"{port_std:.4f}")
    
    with col2:
        if benchmark_data:
            bench_mean = sum(benchmark_data) / len(benchmark_data)
            bench_std = (sum([(x - bench_mean)**2 for x in benchmark_data]) / len(benchmark_data))**0.5
            
            st.markdown("**Benchmark**")
            st.metric("Mean Return", f"{bench_mean:.4f}")
            st.metric("Volatility", f"{bench_std:.4f}")
    
    with col3:
        if show_active and active_data:
            active_mean = sum(active_data) / len(active_data)
            active_std = (sum([(x - active_mean)**2 for x in active_data]) / len(active_data))**0.5
            
            st.markdown("**Active**")
            st.metric("Mean Return", f"{active_mean:.4f}")
            st.metric("Tracking Error", f"{active_std:.4f}")

def render_multi_component_picker(data_loader, sidebar_state):
    """Multi-panel picker for child components (limit 5)"""
    
    st.subheader("Component Comparison")
    
    current_node = sidebar_state.selected_node
    
    # Get child components
    children = data_loader.get_drilldown_options(current_node)
    
    if not children:
        st.info(f"No child components available for {current_node}")
        return
    
    # Component selector (limit to 5)
    max_components = min(5, len(children))
    
    st.markdown(f"**Select up to {max_components} components to compare:**")
    
    selected_components = st.multiselect(
        "Components",
        options=children,
        default=children[:max_components] if len(children) <= max_components else [],
        max_selections=max_components,
        key="component_comparison_selector"
    )
    
    if not selected_components:
        st.info("Select components to view comparison")
        return
    
    # Create comparison chart
    fig = go.Figure()
    
    for i, component in enumerate(selected_components):
        # Get time series for this component
        component_data = data_loader.get_time_series_data('portfolio_returns', component)
        
        if component_data:
            # Apply date range filter
            start_idx, end_idx = sidebar_state.date_range
            if start_idx > 0 or end_idx < len(component_data) - 1:
                component_data = component_data[start_idx:end_idx + 1]
            
            periods = list(range(len(component_data)))
            
            # Use different colors for each component
            color = px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
            
            fig.add_trace(go.Scatter(
                x=periods,
                y=component_data,
                mode='lines',
                name=component,
                line=dict(color=color, width=2),
                hovertemplate=f'{component}<br>Period: %{{x}}<br>Return: %{{y:.4f}}<extra></extra>'
            ))
    
    if fig.data:
        fig.update_layout(
            title="Component Returns Comparison",
            xaxis_title="Period",
            yaxis_title="Return",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for selected components")

def render_rolling_statistics(data_loader, sidebar_state):
    """Render rolling statistics (computed later - placeholder for now)"""
    
    st.subheader("Rolling Statistics")
    
    st.markdown("""
    **Rolling Statistics Analysis** (Coming Soon)
    
    This section will provide:
    - Rolling volatility (21/63/252 day windows)
    - Rolling tracking error
    - Rolling correlation analysis
    - Rolling Sharpe ratios
    
    These metrics will be computed from the time series data and displayed
    with interactive charts showing how risk characteristics evolve over time.
    """)
    
    # Placeholder selectors
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox(
            "Rolling Window",
            options=["21 days", "63 days", "252 days"],
            index=1,
            key="rolling_window_selector",
            disabled=True
        )
    
    with col2:
        st.selectbox(
            "Rolling Metric",
            options=["Volatility", "Tracking Error", "Correlation", "Sharpe Ratio"],
            index=0,
            key="rolling_metric_selector",
            disabled=True
        )
    
    st.info("Rolling statistics will be computed from time_series.* data when available")
    
    # Show what time series data is available
    current_node = sidebar_state.selected_node
    available_series = []
    
    portfolio_data = data_loader.get_time_series_data('portfolio_returns', current_node)
    if portfolio_data:
        available_series.append(f"Portfolio returns: {len(portfolio_data)} periods")
    
    benchmark_data = data_loader.get_time_series_data('benchmark_returns', current_node)
    if benchmark_data:
        available_series.append(f"Benchmark returns: {len(benchmark_data)} periods")
    
    active_data = data_loader.get_time_series_data('active_returns', current_node)
    if active_data:
        available_series.append(f"Active returns: {len(active_data)} periods")
    
    if available_series:
        st.markdown("**Available Time Series:**")
        for series in available_series:
            st.markdown(f"• {series}")
    else:
        st.info("No time series data available for rolling statistics")