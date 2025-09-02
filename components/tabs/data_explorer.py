"""
Data Explorer Tab for Maverick UI

This module provides comprehensive data exploration capabilities including:
- Time series returns visualization with compounding options
- Interactive scatter plot analysis of nodes vs factors
- Multi-lens analysis (portfolio, benchmark, active)

Features:
- First row: Two line chart columns with compounding toggles
- Second row: Interactive scatter plot with rug plots and hover information
- Professional formatting with percentage axes
- Multi-node selection capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import sys
import os

# Add parent directories for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.formatters import format_percentage


def render_data_explorer_tab(data_access_service, sidebar_state):
    """
    Main render function for Data Explorer tab.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        The data access service instance providing data access
    sidebar_state : SidebarState
        Current sidebar filter selections
    """
    st.header("Data Explorer")
    
    component_id = sidebar_state.selected_component_id
    
    # First row: Two columns with line charts and toggles
    st.subheader("Returns Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_descendant_returns_chart(data_access_service, component_id)
    
    with col2:
        render_factor_returns_chart(data_access_service)
    
    st.divider()
    
    # Second row: Interactive scatter plot
    st.subheader("Correlation Analysis")
    render_scatter_plot_analysis(data_access_service, sidebar_state)


def render_descendant_returns_chart(data_access_service, component_id: str):
    """
    Render descendant returns line chart with compounding toggle.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        Data access service instance
    component_id : str
        Selected component ID
    """
    st.markdown("**Descendant Returns**")
    
    # Toggle for compounding
    compound_descendants = st.checkbox(
        "Show Compounded Returns",
        value=False,
        key="compound_descendants"
    )
    
    # Lens selection for descendants
    lens_descendants = st.selectbox(
        "Lens",
        ["portfolio", "benchmark", "active"],
        index=0,
        key="lens_descendants"
    )
    
    try:
        # Get descendant returns data
        descendant_data = data_access_service.get_descendant_returns_data(component_id, lens_descendants)
        
        if descendant_data.empty:
            st.info(f"No descendant returns data available for {component_id} ({lens_descendants} lens)")
            return
        
        # Apply compounding if requested
        if compound_descendants:
            # Convert to cumulative returns
            plot_data = (1 + descendant_data).cumprod() - 1
            y_label = "Cumulative Returns"
        else:
            plot_data = descendant_data
            y_label = "Periodic Returns"
        
        # Create line chart
        fig = px.line(
            plot_data,
            title=f"{lens_descendants.title()} Returns - {len(plot_data.columns)} Components",
            labels={"value": y_label, "index": "Date"}
        )
        
        # Format y-axis as percentages
        
        # Improve layout
        fig.update_layout(
            yaxis=dict(tickformat=".1%"),
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary stats
        st.caption(f"Showing {len(plot_data.columns)} descendant components")
        
    except Exception as e:
        st.error(f"Error loading descendant returns: {e}")


def render_factor_returns_chart(data_access_service):
    """
    Render factor returns line chart with compounding toggle.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        Data access service instance
    """
    st.markdown("**Factor Returns**")
    
    # Toggle for compounding
    compound_factors = st.checkbox(
        "Show Compounded Returns",
        value=False,
        key="compound_factors"
    )
    
    try:
        # Get factor returns data
        factor_data = data_access_service.get_factor_returns_data()
        
        if factor_data.empty:
            st.info("No factor returns data available")
            return
        
        # Factor selection for display (limit to avoid overcrowding)
        available_factors = list(factor_data.columns)
        
        # Default to first 10 factors if many available
        default_factors = available_factors[:10] if len(available_factors) > 10 else available_factors
        
        selected_factors = st.multiselect(
            "Select Factors to Display",
            available_factors,
            default=default_factors,
            key="selected_factors_chart"
        )
        
        if not selected_factors:
            st.info("Please select at least one factor to display")
            return
        
        # Filter to selected factors
        display_data = factor_data[selected_factors]
        
        # Apply compounding if requested
        if compound_factors:
            # Convert to cumulative returns
            plot_data = (1 + display_data).cumprod() - 1
            y_label = "Cumulative Returns"
        else:
            plot_data = display_data
            y_label = "Periodic Returns"
        
        # Create line chart
        fig = px.line(
            plot_data,
            title=f"Factor Returns - {len(selected_factors)} Factors",
            labels={"value": y_label, "index": "Date"}
        )
        
        # Format y-axis as percentages
        
        # Improve layout
        fig.update_layout(
            yaxis=dict(tickformat=".1%"),
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary stats
        st.caption(f"Showing {len(selected_factors)} of {len(available_factors)} available factors")
        
    except Exception as e:
        st.error(f"Error loading factor returns: {e}")


def render_scatter_plot_analysis(data_access_service, sidebar_state):
    """
    Render interactive scatter plot with multi-node selection vs factor analysis.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        Data access service instance
    sidebar_state : SidebarState
        Current sidebar state
    """
    st.markdown("**Node vs Factor Analysis**")
    
    # Get available components and factors
    component_id = sidebar_state.selected_component_id
    
    try:
        # Get descendant components for node selection
        descendant_ids = data_access_service.get_descendant_leaf_ids(component_id)
        available_factors = data_access_service.get_factor_list()
        
        if not descendant_ids:
            st.info(f"No descendant components found for {component_id}")
            return
            
        if not available_factors:
            st.info("No factors available for analysis")
            return
        
        # User selections
        col1, col2 = st.columns(2)
        
        with col1:
            # Multi-select dropdown for nodes
            selected_nodes = st.selectbox(
                "Select Nodes (Y-axis)",
                ["Select multiple..."] + descendant_ids,
                key="nodes_dropdown_trigger"
            )
            
            # Show multiselect when dropdown is used
            if selected_nodes != "Select multiple...":
                # If single node selected from dropdown, use it
                selected_nodes = [selected_nodes]
            else:
                # Show full multiselect interface
                selected_nodes = st.multiselect(
                    "Choose multiple nodes:",
                    descendant_ids,
                    default=descendant_ids[:5] if len(descendant_ids) > 5 else descendant_ids,
                    key="selected_nodes_scatter"
                )
            
            # Lens selection for nodes
            node_lens = st.selectbox(
                "Node Lens",
                ["portfolio", "benchmark", "active"],
                index=0,
                key="node_lens_scatter"
            )
        
        with col2:
            # Single select for factor (x-axis)
            selected_factor = st.selectbox(
                "Select Factor (X-axis)",
                available_factors,
                key="selected_factor_scatter"
            )
        
        if not selected_nodes or not selected_factor:
            st.info("Please select nodes and a factor for analysis")
            return
        
        # Get data for scatter plot
        scatter_data = prepare_scatter_plot_data(
            data_access_service, selected_nodes, selected_factor, node_lens
        )
        
        if scatter_data.empty:
            st.warning("No data available for selected nodes and factor")
            return
        
        # Create scatter plot
        render_scatter_plot(scatter_data, selected_factor, node_lens, selected_nodes)
        
    except Exception as e:
        st.error(f"Error creating scatter plot analysis: {e}")


def prepare_scatter_plot_data(data_access_service, node_ids: List[str], factor_name: str, lens: str) -> pd.DataFrame:
    """
    Prepare data for scatter plot analysis.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        Data access service instance
    node_ids : List[str]
        List of node component IDs
    factor_name : str
        Factor name for x-axis
    lens : str
        Lens type for node returns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with node returns and factor returns aligned
    """
    try:
        # Get factor returns
        factor_data = data_access_service.get_factor_returns_data([factor_name])
        if factor_data.empty:
            return pd.DataFrame()
        
        factor_returns = factor_data[factor_name]
        
        # Collect node returns data
        scatter_records = []
        
        for node_id in node_ids:
            # Get node returns based on lens
            if lens == "portfolio":
                node_returns = data_access_service.get_portfolio_returns(node_id)
            elif lens == "benchmark":
                node_returns = data_access_service.get_benchmark_returns(node_id)
            elif lens == "active":
                node_returns = data_access_service.get_active_returns(node_id)
            else:
                continue
            
            if node_returns.empty:
                continue
            
            # Align factor and node returns
            aligned_factor, aligned_node = factor_returns.align(node_returns, join='inner')
            
            if aligned_factor.empty or aligned_node.empty:
                continue
            
            # Create records for scatter plot
            for date, (factor_val, node_val) in zip(aligned_factor.index, zip(aligned_factor.values, aligned_node.values)):
                if not (pd.isna(factor_val) or pd.isna(node_val)):
                    scatter_records.append({
                        'date': date,
                        'factor_return': factor_val,
                        'node_return': node_val,
                        'node_id': node_id,
                        'factor_name': factor_name
                    })
        
        return pd.DataFrame(scatter_records)
        
    except Exception as e:
        st.error(f"Error preparing scatter plot data: {e}")
        return pd.DataFrame()


def render_scatter_plot(data: pd.DataFrame, factor_name: str, lens: str, node_ids: List[str]):
    """
    Render the scatter plot with rug plots and hover information.
    
    Parameters
    ----------
    data : pd.DataFrame
        Scatter plot data
    factor_name : str
        Factor name for x-axis
    lens : str
        Lens type for node returns
    node_ids : List[str]
        List of selected node IDs
    """
    # Create scatter plot with marginal rugs
    fig = px.scatter(
        data,
        x='factor_return',
        y='node_return',
        color='node_id',
        title=f"{lens.title()} Returns vs {factor_name}",
        labels={
            'factor_return': f'{factor_name} Returns',
            'node_return': f'{lens.title()} Returns'
        },
        hover_data={
            'date': True,
            'factor_return': ':.1%',
            'node_return': ':.1%'
        },
        marginal_x="rug",
        marginal_y="rug"
    )
    
    # Format axes as percentages with 1 decimal
    fig.update_layout(
        xaxis=dict(tickformat=".1%"),
        yaxis=dict(tickformat=".1%")
    )
    
    # Add trend line if multiple points
    if len(data) > 10:
        # Add trend line for each node
        for node_id in data['node_id'].unique():
            node_data = data[data['node_id'] == node_id]
            if len(node_data) > 5:  # Only add trend line if enough points
                z = np.polyfit(node_data['factor_return'], node_data['node_return'], 1)
                p = np.poly1d(z)
                
                x_trend = np.linspace(node_data['factor_return'].min(), node_data['factor_return'].max(), 100)
                y_trend = p(x_trend)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        name=f'{node_id} trend',
                        line=dict(dash='dash'),
                        opacity=0.7,
                        showlegend=False
                    )
                )
    
    # Improve layout
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Points", len(data))
    
    with col2:
        correlation = data['factor_return'].corr(data['node_return'])
        st.metric("Overall Correlation", f"{correlation:.3f}")
    
    with col3:
        st.metric("Nodes Analyzed", len(node_ids))
    
    # Show per-node correlations
    with st.expander("Per-Node Correlations"):
        node_correlations = []
        for node_id in data['node_id'].unique():
            node_data = data[data['node_id'] == node_id]
            if len(node_data) > 2:
                corr = node_data['factor_return'].corr(node_data['node_return'])
                node_correlations.append({
                    'Node': node_id,
                    'Correlation': f"{corr:.3f}",
                    'Data Points': len(node_data)
                })
        
        if node_correlations:
            corr_df = pd.DataFrame(node_correlations)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)