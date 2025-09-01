"""
Risk Decomposition Tab for Maverick UI

This module provides comprehensive risk decomposition analysis using the simplified
risk API with direct RiskResult access for optimal performance, displaying:
1. Risk summary KPIs (total volatility, factor/specific/cross-correlation contributions)  
2. Factor analysis (exposures and contributions)
3. Component analysis table (weights and risk contributions)
4. Risk matrices (weighted betas and factor risk contributions)

For active risk analysis, displays all four components:
- Factor risk contribution (summable)
- Specific risk contribution (summable)  
- Cross-correlation risk contribution (summable)
- Total active risk (Euler identity validated)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sys
import os

# Add parent directories for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.formatters import format_basis_points, format_percentage, format_decimal, truncate_component_name
from utils.colors import get_chart_color

def render_risk_decomposition_tab(data_access_service, sidebar_state):
    """
    Main render function for Risk Decomposition tab.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        The data access service instance providing schema access
    sidebar_state : SidebarState
        Current sidebar filter selections
    """
    lens = sidebar_state.lens

    st.header("Risk Decomposition")
    st.markdown(f"**Component:** {sidebar_state.selected_component_id} | **Lens:** {lens.title()}")
    
    # Data availability diagnostics
    # render_data_availability_diagnostics(data_access_service, sidebar_state)
    
    # Extract risk decomposition data for current selection
    risk_data = data_access_service.get_risk_decomposition(sidebar_state.selected_component_id, lens)
    
    # Debug: Show available risk data fields for development
    if st.checkbox("Show debug info (available risk data fields)"):
        st.json({k: type(v).__name__ for k, v in risk_data.items() if risk_data})
    
    #if not risk_data:
        #st.warning(f"Risk decomposition data not available for {sidebar_state.selected_component_id} ({sidebar_state.lens} lens)")
        #st.info("This could mean risk analysis has not been run for this component/lens combination.")
        #return
    
    # Row 1: Risk Summary KPIs
    render_risk_summary_kpis(risk_data)
    
    st.divider()
    

# =====================================================================
# ROW 1: RISK SUMMARY KPIs
# =====================================================================

def render_risk_summary_kpis(risk_data: Dict[str, Any]):
    """
    Render Row 1: Risk summary KPIs showing complete volatility breakdown.
    
    Parameters
    ----------
    risk_data : Dict[str, Any]
        Risk data extracted from schema
    """
    st.subheader("Risk Summary")
    
    # Extract risk metrics
    total_risk = risk_data.get('total_risk', 0)
    factor_risk_contrib = risk_data.get('factor_risk_contribution', 0)
    specific_risk_contrib = risk_data.get('specific_risk_contribution', 0)
    cross_corr_contrib = risk_data.get('cross_correlation_risk_contribution', 0)
    
    # Check if we have cross-correlation (active risk) or just portfolio risk
    has_cross_correlation = abs(cross_corr_contrib) > 1e-10
    
    if has_cross_correlation:
        # Active risk: show all four components
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Active Risk", 
                format_basis_points(total_risk),
                help="Total active risk (tracking error volatility)"
            )
        
        with col2:
            st.metric(
                "Factor Risk Contribution", 
                format_basis_points(factor_risk_contrib),
                help="Risk contribution from factor exposures"
            )
        
        with col3:
            st.metric(
                "Idiosyncratic Risk Contribution", 
                format_basis_points(specific_risk_contrib),
                help="Risk contribution from asset-specific factors"
            )
            
        with col4:
            st.metric(
                "Cross-Correlation Contribution", 
                format_basis_points(cross_corr_contrib),
                help="Risk contribution from cross-correlation between benchmark and active returns"
            )
        
        # Complete validation check for active risk
        sum_contributions = factor_risk_contrib + specific_risk_contrib + cross_corr_contrib
        if abs(sum_contributions - total_risk) > 1e-6:
            st.warning(f"Risk decomposition validation: Factor + Idiosyncratic + Cross-Correlation ({format_basis_points(sum_contributions)}) ≠ Total ({format_basis_points(total_risk)})")
        else:
            st.success("✓ Euler identity validated: All risk contributions sum to total risk")
    
    else:
        # Portfolio risk: show traditional three components
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Portfolio Risk", 
                format_basis_points(total_risk),
                help="Total portfolio risk (volatility)"
            )
        
        with col2:
            st.metric(
                "Factor Risk Contribution", 
                format_basis_points(factor_risk_contrib),
                help="Risk contribution from factor exposures"
            )
        
        with col3:
            st.metric(
                "Idiosyncratic Risk Contribution", 
                format_basis_points(specific_risk_contrib),
                help="Risk contribution from asset-specific factors"
            )
        
        # Portfolio validation check
        sum_contributions = factor_risk_contrib + specific_risk_contrib
        if abs(sum_contributions - total_risk) > 1e-6:
            st.warning(f"Risk decomposition validation: Factor + Idiosyncratic ({format_basis_points(sum_contributions)}) ≠ Total ({format_basis_points(total_risk)})")
        else:
            st.success("✓ Euler identity validated: All risk contributions sum to total risk")




# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def extract_risk_decomposition_data(data_access_service, component_id: str, lens: str) -> Dict[str, Any]:
    """
    Central data extraction function for risk decomposition.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        The data access service instance
    component_id : str
        Component ID to analyze
    lens : str
        Lens perspective (portfolio, benchmark, active)
        
    Returns
    -------
    Dict[str, Any]
        Extracted risk decomposition data
    """
    try:
        return data_access_service.get_risk_decomposition(component_id, lens)
    except Exception as e:
        return {}


def get_child_components(data_access_service, component_id: str) -> List[str]:
    """
    Get child components for a given component.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        The data access service instance
    component_id : str
        Parent component ID
        
    Returns
    -------
    List[str]
        List of child component IDs
    """
    # Get all components and filter for children
    try:
        all_components = list(data_access_service.risk_analysis_service._portfolio_graph.components.keys()) if data_access_service.risk_analysis_service._portfolio_graph else []
    except:
        all_components = []
    
    # Simple heuristic: components that start with the parent component path + "/"
    child_pattern = f"{component_id}/"
    children = [comp for comp in all_components if comp.startswith(child_pattern)]
    
    # Filter to direct children only (not grandchildren)
    direct_children = []
    for child in children:
        # Remove parent prefix and check if there are additional levels
        relative_path = child[len(child_pattern):]
        if '/' not in relative_path:  # Direct child
            direct_children.append(child)
    
    return sorted(direct_children)


def sort_and_filter_factors(factor_dict: Dict[str, float], top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Sort factors by absolute value and return top N.
    
    Parameters
    ----------
    factor_dict : Dict[str, float]
        Factor name to value mapping
    top_n : int
        Number of top factors to return
        
    Returns
    -------
    List[Tuple[str, float]]
        Sorted list of (factor_name, value) tuples
    """
    if not factor_dict:
        return []
    
    return sorted(factor_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]


def create_risk_matrix_heatmap(data: List[List[float]], x_labels: List[str], y_labels: List[str], 
                              title: str, color_scale: str = "RdBu", center_colorscale: bool = True) -> go.Figure:
    """
    Create a heatmap visualization for risk matrices.
    
    Parameters
    ----------
    data : List[List[float]]
        2D matrix data
    x_labels : List[str]
        X-axis labels (factors)
    y_labels : List[str]  
        Y-axis labels (components)
    title : str
        Chart title
    color_scale : str
        Plotly color scale name
    center_colorscale : bool
        Whether to center the color scale around zero
        
    Returns
    -------
    go.Figure
        Plotly heatmap figure
    """
    if not data or not data[0]:
        return go.Figure()
    
    # Convert to numpy array for easier handling
    data_array = np.array(data)
    
    # Configure color scale centering
    zmid = 0 if center_colorscale else None
    
    fig = go.Figure(data=go.Heatmap(
        z=data_array,
        x=x_labels,
        y=y_labels,
        colorscale=color_scale,
        zmid=zmid,
        hoverongaps=False,
        colorbar=dict(title="Value")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Factors",
        yaxis_title="Components",
        height=max(400, len(y_labels) * 25 + 150),
        font=dict(size=10)
    )
    
    return fig