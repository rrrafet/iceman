"""
Allocation-Selection Tab for Maverick UI

This module provides Brinson-style allocation and selection risk decomposition analysis
for a selected node in a hierarchical portfolio. Shows a sortable table of all descendant 
leaves (assets) with comprehensive active risk columns.

Features:
- 12-column table format as specified in requirements
- Support for both active risk (Brinson decomposition) and portfolio risk analysis
- Asset-level allocation and selection risk breakdowns
- CSV export with semicolon delimiter
- Robust handling of missing data
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# Add parent directories for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.formatters import format_basis_points, format_percentage, format_decimal


def render_allocation_selection_tab(data_access_service, sidebar_state):
    """
    Main render function for Allocation-Selection tab.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        The data access service instance providing data access
    sidebar_state : SidebarState
        Current sidebar filter selections
    """
    st.header("Allocation-Selection")
    
    component_id = sidebar_state.selected_component_id
    lens = sidebar_state.lens
    
    # Extract risk decomposition data for current selection
    risk_data = extract_allocation_selection_data(data_access_service, component_id, lens)
    
    if not risk_data or "error" in risk_data:
        st.warning(f"Risk decomposition data not available for {component_id} ({lens} lens)")
        st.info("This could mean risk analysis has not been run for this component/lens combination.")
        return
    
    # Build descendant leaf asset list
    leaf_ids = build_leaf_asset_list(data_access_service, component_id)
    
    if not leaf_ids:
        st.warning(f"No descendant leaf assets found for component {component_id}")
        st.info("This tab shows asset-level analysis for leaf nodes. Try selecting a component that has asset descendants.")
        return
    
    # Render header with component info and badges
    render_header_badges(component_id, risk_data, len(leaf_ids))
    
    # Create Brinson analysis table
    brinson_df = data_access_service.riskresult_to_brinson_table(risk_data, leaf_ids)
    
    if brinson_df.empty:
        st.warning("No data available to create allocation-selection table")
        return
    
    # Render the analysis table
    render_brinson_analysis_table(brinson_df, risk_data)
    
    # Add CSV download functionality
    create_csv_download(brinson_df, component_id, lens)


def extract_allocation_selection_data(data_access_service, component_id: str, lens: str) -> Optional[Dict[str, Any]]:
    """
    Extract comprehensive risk decomposition data specifically for allocation-selection analysis.
    
    Uses the specialized get_allocation_selection_decomposition method that provides
    ALL necessary fields for the 12-column Brinson table, including allocation and
    selection components that are missing from the general get_risk_decomposition method.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        Data access service instance
    component_id : str
        Component ID to analyze
    lens : str
        Risk lens ('portfolio', 'benchmark', 'active')
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Comprehensive risk decomposition data or None if not available
    """
    try:
        risk_data = data_access_service.get_allocation_selection_decomposition(component_id, lens)
        return risk_data
    except Exception as e:
        st.error(f"Error extracting allocation-selection risk data: {e}")
        return None


def build_leaf_asset_list(data_access_service, component_id: str) -> list[str]:
    """
    Build list of descendant leaf asset IDs for the selected component.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        Data access service instance
    component_id : str
        Component ID to get descendants for
        
    Returns
    -------
    list[str]
        List of leaf component IDs
    """
    try:
        leaf_ids = data_access_service.get_descendant_leaf_ids(component_id)
        return leaf_ids
    except Exception as e:
        st.error(f"Error getting descendant leaves: {e}")
        return []


def render_header_badges(component_id: str, risk_data: Dict[str, Any], num_assets: int):
    """
    Render header section with component info and analysis badges.
    
    Parameters
    ----------
    component_id : str
        Selected component ID
    risk_data : Dict[str, Any]
        Risk decomposition data
    num_assets : int
        Number of descendant assets
    """
    st.markdown(f"**Selected Component:** {component_id}")
    st.markdown(f"**Descendant Assets:** {num_assets}")
    
    # Create badges for analysis metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_type = risk_data.get('analysis_type', 'portfolio')
        st.info(f"Analysis: {analysis_type.title()}")
    
    with col2:
        frequency = risk_data.get('frequency', 'Unknown')
        st.info(f"Frequency: {frequency}")
    
    with col3:
        annualized = risk_data.get('annualized', False)
        st.info(f"Annualized: {'Yes' if annualized else 'No'}")


def render_brinson_analysis_table(df: pd.DataFrame, risk_data: Dict[str, Any]):
    """
    Render the main Brinson-style allocation-selection table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Brinson table with 12 columns
    risk_data : Dict[str, Any]
        Risk decomposition data for context
    """
    st.subheader(f"Asset-Level Analysis ({len(df)} Assets)")
    
    # Show analysis type context
    analysis_type = risk_data.get('analysis_type', 'portfolio')
    if analysis_type == "active":
        st.info("🎯 **Active Risk Analysis**: Allocation and Selection columns show Brinson-style risk decomposition")
    else:
        st.info("📊 **Portfolio Risk Analysis**: Allocation and Selection columns are zero (portfolio-only lens)")
    
    # Configure column formatting for display
    formatted_df = format_table_for_display(df)
    
    # Display the sortable table
    st.dataframe(
        formatted_df,
        use_container_width=True,
        hide_index=True,
        height=min(600, (len(formatted_df) + 1) * 35 + 3)
    )
    
    # Show summary statistics
    render_table_summary_stats(df, analysis_type)


def format_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the Brinson table for optimal Streamlit display.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw Brinson table
        
    Returns
    -------
    pd.DataFrame
        Formatted table for display
    """
    display_df = df.copy()
    
    # Format numeric columns with specific formatting requirements
    weight_columns = ['portfolio_weight', 'benchmark_weight']
    contribution_columns = [
        'asset contribution to risk', 'asset contribution to factor risk',
        'asset contribution to specific risk', 'asset contribution to allocation factor risk',
        'asset contribution to allocation specific risk', 'asset contribution to selection factor risk',
        'asset contribution to selection specific risk'
    ]
    marginal_columns = ['asset marginal']
    
    # Format weight columns as percentages with 1 decimal place
    for col in weight_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x * 100:.1f}%" if not pd.isna(x) else "NaN"
            )
    
    # Format contribution columns as basis points with 0 decimal places
    for col in contribution_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x * 10000:.0f} bps" if not pd.isna(x) else "NaN"
            )
    
    # Format marginal columns with 2 decimal places (unchanged units)
    for col in marginal_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x * 10000:.0f}" if not pd.isna(x) else "NaN"
            )
    
    return display_df


def render_table_summary_stats(df: pd.DataFrame, analysis_type: str):
    """
    Render summary statistics for the Brinson table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Brinson table
    analysis_type : str
        Type of analysis ('active' or 'portfolio')
    """
    st.subheader("Summary Statistics")
    
    # Calculate summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_port_weight = df['portfolio_weight'].sum() if not df['portfolio_weight'].isna().all() else 0
        st.metric("Total Portfolio Weight", f"{total_port_weight:.4f}")
    
    with col2:
        total_bench_weight = df['benchmark_weight'].sum() if not df['benchmark_weight'].isna().all() else 0
        st.metric("Total Benchmark Weight", f"{total_bench_weight:.4f}")
    
    with col3:
        total_risk_contrib = df['asset contribution to risk'].sum() if not df['asset contribution to risk'].isna().all() else 0
        st.metric("Total Risk Contribution", format_basis_points(total_risk_contrib))
    
    with col4:
        total_factor_risk = df['asset contribution to factor risk'].sum() if not df['asset contribution to factor risk'].isna().all() else 0
        st.metric("Total Factor Risk", format_basis_points(total_factor_risk))
    
    # Show allocation/selection summary if active analysis
    if analysis_type == "active":
        st.markdown("**Allocation vs Selection Breakdown**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            alloc_factor = df['asset contribution to allocation factor risk'].sum()
            st.metric("Allocation Factor", format_basis_points(alloc_factor))
        
        with col2:
            alloc_specific = df['asset contribution to allocation specific risk'].sum()
            st.metric("Allocation Specific", format_basis_points(alloc_specific))
        
        with col3:
            select_factor = df['asset contribution to selection factor risk'].sum()
            st.metric("Selection Factor", format_basis_points(select_factor))
        
        with col4:
            select_specific = df['asset contribution to selection specific risk'].sum()
            st.metric("Selection Specific", format_basis_points(select_specific))


def create_csv_download(df: pd.DataFrame, component_id: str, lens: str):
    """
    Create CSV download functionality with semicolon delimiter.
    
    Parameters
    ----------
    df : pd.DataFrame
        Brinson table to export
    component_id : str
        Selected component ID
    lens : str
        Risk lens
    """
    st.subheader("Export")
    
    if st.button("Download CSV", type="primary"):
        # Export with semicolon delimiter and UTF-8 encoding as specified
        csv_data = df.to_csv(sep=';', index=False, encoding='utf-8')
        
        # Create download filename
        filename = f"allocation_selection_{component_id.replace('/', '_')}_{lens}.csv"
        
        st.download_button(
            label="Download Allocation-Selection Table",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            help="Download the complete 12-column table with semicolon delimiter"
        )
    
    # Show export info
    st.caption(f"Export format: {len(df)} rows × {len(df.columns)} columns, semicolon-delimited, UTF-8 encoding")