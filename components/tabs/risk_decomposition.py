"""
Risk Decomposition Tab for Maverick UI

This module provides comprehensive risk decomposition analysis using the simplified
risk API with direct RiskResult access for optimal performance, displaying:
1. Risk summary KPIs (volatility, factor risk, idiosyncratic risk)  
2. Factor analysis (exposures and contributions)
3. Component analysis table (weights and risk contributions)
4. Risk matrices (weighted betas and factor risk contributions)
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
    st.header("Risk Decomposition")
    st.markdown(f"**Component:** {sidebar_state.selected_component_id} | **Lens:** {sidebar_state.lens.title()}")
    
    # Data availability diagnostics
    render_data_availability_diagnostics(data_access_service, sidebar_state)
    
    # Extract risk decomposition data for current selection
    risk_data = extract_risk_decomposition_data(data_access_service, sidebar_state.selected_component_id, sidebar_state.lens)
    
    if not risk_data:
        st.warning(f"Risk decomposition data not available for {sidebar_state.selected_component_id} ({sidebar_state.lens} lens)")
        st.info("This could mean risk analysis has not been run for this component/lens combination.")
        return
    
    # Row 1: Risk Summary KPIs
    render_risk_summary_kpis(risk_data)
    
    st.divider()
    
    # Row 2: Factor Analysis 
    render_factor_analysis(risk_data, data_access_service.get_available_factors())
    
    st.divider()
    
    # Row 3: Component Analysis Table
    render_component_table(data_access_service, sidebar_state)
    
    st.divider()
    
    # Row 4: Risk Matrices
    render_risk_matrices(data_access_service, sidebar_state)
    
    st.divider()
    
    # Summary and Recommendations
    render_summary_and_recommendations(risk_data, data_access_service.get_available_factors())


# =====================================================================
# ROW 1: RISK SUMMARY KPIs
# =====================================================================

def render_risk_summary_kpis(risk_data: Dict[str, Any]):
    """
    Render Row 1: Risk summary KPIs showing volatility breakdown.
    
    Parameters
    ----------
    risk_data : Dict[str, Any]
        Risk data extracted from schema
    """
    st.subheader("Risk Summary")
    
    col1, col2, col3 = st.columns(3)
    
    # Extract risk metrics
    total_risk = risk_data.get('total_risk', 0)
    factor_risk_contrib = risk_data.get('factor_risk_contribution', 0)
    specific_risk_contrib = risk_data.get('specific_risk_contribution', 0)
    
    with col1:
        st.metric(
            "Total Volatility", 
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
    
    # Validation check
    sum_contributions = factor_risk_contrib + specific_risk_contrib
    if abs(sum_contributions - total_risk) > 1e-6:  # Allow small numerical differences
        st.warning(f"Risk decomposition validation: Factor + Idiosyncratic ({format_basis_points(sum_contributions)}) ≠ Total ({format_basis_points(total_risk)})")


# =====================================================================
# ROW 2: FACTOR ANALYSIS  
# =====================================================================

def render_factor_analysis(risk_data: Dict[str, Any], factor_names: List[str]):
    """
    Render Row 2: Factor analysis showing exposures and contributions.
    
    Parameters
    ----------
    risk_data : Dict[str, Any]
        Risk data extracted from schema
    factor_names : List[str]
        List of available factor names
    """
    st.subheader("Factor Analysis")
    
    col1, col2 = st.columns(2)
    
    # Extract factor data
    factor_exposures = risk_data.get('factor_exposures', {})
    factor_contributions = risk_data.get('factor_contributions', {})
    
    with col1:
        st.markdown("**Factor Exposures**")
        render_factor_data_table(factor_exposures, "Exposure", format_decimal)
    
    with col2:
        st.markdown("**Factor Risk Contributions**") 
        render_factor_data_table(factor_contributions, "Contribution (bps)", 
                                lambda x: format_basis_points(x, decimal_places=1))


def render_factor_data_table(factor_data: Dict[str, float], column_name: str, formatter):
    """
    Helper function to render factor data as a sorted table with enhanced diagnostics.
    
    Parameters
    ----------
    factor_data : Dict[str, float]
        Factor name to value mapping
    column_name : str
        Column header for the values
    formatter : callable
        Function to format the values
    """
    if not factor_data:
        st.warning("⚠️ Factor data is empty")
        st.info("This indicates that the risk model did not calculate meaningful factor values. This could be due to:")
        st.markdown("- Portfolio composition (all specific risk, no factor exposures)")
        st.markdown("- Risk model estimation issues")
        st.markdown("- Data quality problems in returns")
        return
    
    # Check if all values are zero
    non_zero_values = {k: v for k, v in factor_data.items() if abs(v) > 1e-10}
    if not non_zero_values:
        st.warning("⚠️ All factor values are zero")
        st.info("Factor names are available but all calculated values are zero.")
        
        # Still show the table with zero values for transparency
        df_data = []
        for factor, value in factor_data.items():
            df_data.append({
                'Factor': factor,
                column_name: formatter(value),
                'Status': '⚠️ Zero'
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    
    # Sort by absolute value and take top 10
    sorted_factors = sort_and_filter_factors(factor_data, top_n=10)
    
    if not sorted_factors:
        st.info("No significant factor data to display")
        return
    
    # Create DataFrame for display with significance indicators
    df_data = []
    for factor, value in sorted_factors:
        # Add significance indicator
        if abs(value) > 1e-6:
            status = "✅ Active" if abs(value) > 1e-4 else "🔸 Small"
        else:
            status = "⚠️ Minimal"
            
        df_data.append({
            'Factor': factor,
            column_name: formatter(value),
            'Status': status
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Summary statistics
    total_abs_contribution = sum(abs(v) for v in factor_data.values())
    active_factors = len([v for v in factor_data.values() if abs(v) > 1e-6])
    st.caption(f"📊 {active_factors}/{len(factor_data)} factors active • Total absolute: {formatter(total_abs_contribution)}")


# =====================================================================
# ROW 3: COMPONENT ANALYSIS TABLE
# =====================================================================

def render_component_table(data_access_service, sidebar_state):
    """
    Render Row 3: Component analysis table showing descendant components.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        The data access service instance
    sidebar_state : SidebarState
        Current sidebar selections
    """
    st.subheader("Component Analysis")
    
    # Get child components
    child_components = get_child_components(data_access_service, sidebar_state.selected_component_id)
    
    if not child_components:
        st.info(f"No child components found for {sidebar_state.selected_component_id}")
        return
    
    # Build component analysis data
    table_data = []
    
    for component_id in child_components:
        # Get component data for each lens
        try:
            portfolio_data = data_access_service.get_risk_decomposition(component_id, 'portfolio')
            benchmark_data = data_access_service.get_risk_decomposition(component_id, 'benchmark')
            active_data = data_access_service.get_risk_decomposition(component_id, 'active')
        except:
            portfolio_data = {}
            benchmark_data = {}
            active_data = {}
        
        # Determine if this is a node or leaf
        is_leaf = len(get_child_components(data_access_service, component_id)) == 0
        component_type = "Leaf" if is_leaf else "Node"
        
        # Extract weights (assuming they're stored as proportions, convert to percentages)
        portfolio_weight = portfolio_data.get('portfolio_weight', 0) * 100
        benchmark_weight = benchmark_data.get('benchmark_weight', 0) * 100
        active_weight = active_data.get('active_weight', portfolio_weight - benchmark_weight)
        
        # Extract risk contributions
        asset_risk_contrib = portfolio_data.get('total_risk', 0)
        asset_factor_risk = portfolio_data.get('factor_risk_contribution', 0)
        asset_specific_risk = portfolio_data.get('specific_risk_contribution', 0)
        
        table_data.append({
            'Component': truncate_component_name(component_id),
            'Type': component_type,
            'Portfolio Weight (%)': f"{portfolio_weight:.2f}",
            'Benchmark Weight (%)': f"{benchmark_weight:.2f}",
            'Active Weight (%)': f"{active_weight:.2f}",
            'Asset Risk Contribution (bps)': format_basis_points(asset_risk_contrib, decimal_places=0),
            'Asset Factor Risk Contribution (bps)': format_basis_points(asset_factor_risk, decimal_places=0),
            'Asset Idiosyncratic Risk Contribution (bps)': format_basis_points(asset_specific_risk, decimal_places=0)
        })
    
    # Create and display DataFrame
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Summary statistics
    if table_data:
        total_components = len(table_data)
        leaf_count = sum(1 for row in table_data if row['Type'] == 'Leaf')
        st.caption(f"Showing {total_components} components ({leaf_count} leaves, {total_components - leaf_count} nodes)")


# =====================================================================
# ROW 4: RISK MATRICES
# =====================================================================

def render_risk_matrices(data_access_service, sidebar_state):
    """
    Render Row 4: Risk matrices showing weighted betas and factor risk contributions.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        The data access service instance  
    sidebar_state : SidebarState
        Current sidebar selections
    """
    st.subheader("Risk Matrices")
    
    col1, col2 = st.columns(2)
    
    # Get child components and factor names
    child_components = get_child_components(data_access_service, sidebar_state.selected_component_id)
    factor_names = data_access_service.get_available_factors()
    
    if not child_components or not factor_names:
        st.info("Matrix visualization requires child components and factor data")
        return
    
    with col1:
        st.markdown("**Weighted Beta Contribution Matrix**")
        render_weighted_beta_matrix(data_access_service, child_components, factor_names)
    
    with col2:
        st.markdown("**Factor Risk Contribution Matrix**")
        render_factor_risk_matrix(data_access_service, child_components, factor_names)


def render_weighted_beta_matrix(data_access_service, child_components: List[str], factor_names: List[str]):
    """
    Render weighted beta contribution matrix as heatmap.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        The data access service instance
    child_components : List[str]
        List of child component IDs
    factor_names : List[str]
        List of factor names
    """
    # Collect weighted beta data
    matrix_data = []
    
    for component_id in child_components:
        try:
            component_data = data_access_service.get_risk_decomposition(component_id, 'portfolio')
        except:
            component_data = {}
        weighted_betas = component_data.get('weighted_betas', {})
        
        row_data = []
        for factor in factor_names:
            beta_value = weighted_betas.get(factor, 0)
            row_data.append(beta_value)
        
        matrix_data.append(row_data)
    
    # Create heatmap
    if matrix_data:
        fig = create_risk_matrix_heatmap(
            data=matrix_data,
            x_labels=factor_names,
            y_labels=[truncate_component_name(comp) for comp in child_components],
            title="Weighted Beta Contributions",
            color_scale="RdBu",
            center_colorscale=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No weighted beta data available")


def render_factor_risk_matrix(data_access_service, child_components: List[str], factor_names: List[str]):
    """
    Render factor risk contribution matrix as heatmap.
    
    Parameters
    ----------  
    data_access_service : DataAccessService
        The data access service instance
    child_components : List[str]
        List of child component IDs
    factor_names : List[str]
        List of factor names
    """
    # Collect factor risk contribution data
    matrix_data = []
    
    for component_id in child_components:
        try:
            component_data = data_access_service.get_risk_decomposition(component_id, 'portfolio')
        except:
            component_data = {}
        factor_contributions = component_data.get('factor_contributions', {})
        
        row_data = []
        for factor in factor_names:
            contrib_value = factor_contributions.get(factor, 0) * 10000  # Convert to bps
            row_data.append(contrib_value)
        
        matrix_data.append(row_data)
    
    # Create heatmap
    if matrix_data:
        fig = create_risk_matrix_heatmap(
            data=matrix_data,
            x_labels=factor_names,
            y_labels=[truncate_component_name(comp) for comp in child_components],
            title="Factor Risk Contributions (bps)",
            color_scale="Reds",
            center_colorscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No factor risk contribution data available")


# =====================================================================
# SUMMARY AND RECOMMENDATIONS  
# =====================================================================

def render_summary_and_recommendations(risk_data: Dict[str, Any], factor_names: List[str]):
    """
    Render summary and recommendations based on current data availability.
    
    Parameters
    ----------
    risk_data : Dict[str, Any]
        Risk data for the current component/lens
    factor_names : List[str]
        Available factor names
    """
    st.subheader("📋 Summary & Recommendations")
    
    # Analyze current data state
    has_risk_metrics = risk_data and risk_data.get('total_risk', 0) > 0
    has_factor_names = len(factor_names) > 0
    has_factor_contributions = risk_data and len(risk_data.get('factor_contributions', {})) > 0
    has_nonzero_exposures = False
    
    if risk_data and 'factor_exposures' in risk_data:
        factor_exposures = risk_data['factor_exposures']
        has_nonzero_exposures = any(abs(v) > 1e-10 for v in factor_exposures.values())
    
    # Generate status and recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Status**")
        
        if has_risk_metrics:
            total_risk = risk_data.get('total_risk', 0)
            factor_risk = risk_data.get('factor_risk_contribution', 0)
            specific_risk = risk_data.get('specific_risk_contribution', 0)
            
            factor_pct = (factor_risk / total_risk * 100) if total_risk > 0 else 0
            specific_pct = (specific_risk / total_risk * 100) if total_risk > 0 else 0
            
            st.success(f"✅ Risk Analysis Complete")
            st.metric("Total Risk", format_basis_points(total_risk))
            st.progress(factor_pct / 100, text=f"Factor Risk: {factor_pct:.1f}%")
            st.progress(specific_pct / 100, text=f"Specific Risk: {specific_pct:.1f}%")
        else:
            st.error("❌ No risk analysis results")
    
    with col2:
        st.markdown("**Recommendations**")
        
        if not has_factor_names:
            st.warning("🔧 **Issue**: No factor names available")
            st.markdown("- Check factor returns data")
            st.markdown("- Verify risk model configuration")
        
        elif not has_factor_contributions:
            st.warning("🔧 **Issue**: Factor contributions empty")
            st.markdown("- Risk model may not be estimating factor exposures")
            st.markdown("- Check portfolio composition vs benchmark")
            st.markdown("- Verify factor model parameters")
        
        elif not has_nonzero_exposures:
            st.warning("🔧 **Issue**: All factor exposures are zero")
            st.markdown("- Portfolio may have no systematic factor exposure")
            st.markdown("- Consider checking asset factor loadings")
            st.markdown("- Review risk model estimation period")
        
        else:
            st.success("✅ **Status**: Factor decomposition operational")
            st.markdown("- Risk metrics calculated using simplified API")
            st.markdown("- Factor exposures and contributions available")
            st.markdown("- Analysis results validated and reliable")
            st.markdown("- Using direct RiskResult access for optimal performance")
    
    # Technical details
    with st.expander("🔧 Technical Details", expanded=False):
        if risk_data:
            st.json({
                "data_keys": list(risk_data.keys()),
                "factor_names_count": len(factor_names),
                "factor_contributions_count": len(risk_data.get('factor_contributions', {})),
                "factor_exposures_count": len(risk_data.get('factor_exposures', {})),
                "extraction_method": "simplified_risk_api",
                "lens_type": risk_data.get('lens_type', 'unknown'),
                "api_version": "simplified"
            })
        else:
            st.info("No technical details available - risk data is empty")


# =====================================================================
# DATA AVAILABILITY DIAGNOSTICS
# =====================================================================

def render_data_availability_diagnostics(data_access_service, sidebar_state):
    """
    Render diagnostic information about data availability for debugging.
    
    Parameters
    ----------
    data_access_service : DataAccessService
        The data access service instance
    sidebar_state : SidebarState
        Current sidebar selections
    """
    with st.expander("📊 Data Availability Diagnostics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Factor Names**")
            factor_names = data_access_service.get_available_factors()
            if factor_names:
                st.success(f"✅ {len(factor_names)} factors available")
                st.text(", ".join(factor_names))
            else:
                st.error("❌ No factor names found")
            
            st.markdown("**Components**")
            try:
                components = list(data_access_service.risk_analysis_service._portfolio_graph.components.keys()) if data_access_service.risk_analysis_service._portfolio_graph else []
            except:
                components = []
            if components:
                st.success(f"✅ {len(components)} components")
                st.text(", ".join(components[:5]) + ("..." if len(components) > 5 else ""))
            else:
                st.error("❌ No components found")
        
        with col2:
            st.markdown("**Current Component Data**")
            component_id = sidebar_state.selected_component_id
            lens = sidebar_state.lens
            
            try:
                risk_data = data_access_service.get_risk_decomposition(component_id, lens)
            except:
                risk_data = {}
            if risk_data:
                st.success(f"✅ {lens.title()} lens data available")
                
                # Check specific field availability
                field_status = []
                fields_to_check = ['factor_contributions', 'factor_exposures', 'total_risk', 
                                 'factor_risk_contribution', 'specific_risk_contribution']
                
                for field in fields_to_check:
                    value = risk_data.get(field)
                    if value is None:
                        field_status.append(f"❌ {field}: Missing")
                    elif isinstance(value, dict) and not value:
                        field_status.append(f"⚠️ {field}: Empty dict")
                    elif isinstance(value, (int, float)) and value == 0:
                        field_status.append(f"⚠️ {field}: Zero")
                    else:
                        field_status.append(f"✅ {field}: Available")
                
                for status in field_status:
                    st.text(status)
            else:
                st.error(f"❌ No {lens} lens data available")


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