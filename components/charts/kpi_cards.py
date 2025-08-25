import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color, COLOR_PALETTE

def render_kpi_card(
    title: str,
    value: float,
    subtitle: str = "",
    format_type: str = "percentage",
    color: str = None
) -> None:
    """Render a single KPI card with value and formatting"""
    
    if color is None:
        color = COLOR_PALETTE["blue"]
    
    # Format value based on type
    if format_type == "percentage":
        formatted_value = f"{value:.2f}%"
    elif format_type == "decimal":
        formatted_value = f"{value:.4f}"
    elif format_type == "basis_points":
        formatted_value = f"{value * 10000:.0f} bps"
    else:
        formatted_value = f"{value:.2f}"
    
    # Create metric card
    st.metric(
        label=title,
        value=formatted_value,
        help=subtitle if subtitle else None
    )

def render_kpi_cards_portfolio_active(
    data_loader,
    sidebar_state
) -> None:
    """Render side-by-side KPI cards for Portfolio and Active"""
    
    # Get data for both lenses
    portfolio_metrics = data_loader.get_core_metrics("portfolio", sidebar_state.selected_node)
    active_metrics = data_loader.get_core_metrics("active", sidebar_state.selected_node)
    
    st.subheader("📊 Key Performance Indicators")
    
    # Two columns for Portfolio | Active
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Portfolio Metrics**")
        
        if portfolio_metrics:
            # Portfolio KPIs
            total_risk = portfolio_metrics.get('total_risk', 0)
            factor_contrib = portfolio_metrics.get('factor_risk_contribution', 0) 
            specific_contrib = portfolio_metrics.get('specific_risk_contribution', 0)
            factor_pct = portfolio_metrics.get('factor_risk_percentage', 0)
            
            # Display metrics
            render_kpi_card("Total Risk", total_risk, "Portfolio total risk", "basis_points")
            render_kpi_card("Factor Risk", factor_contrib, "Factor contribution", "basis_points")
            render_kpi_card("Specific Risk", specific_contrib, "Specific contribution", "basis_points") 
            render_kpi_card("Factor %", factor_pct, "Factor risk percentage", "percentage")
        else:
            st.info("Portfolio metrics not available for selected node")
    
    with col2:
        st.markdown("**Active Metrics**")
        
        if active_metrics:
            # Active KPIs  
            active_risk = active_metrics.get('total_risk', 0)
            active_factor = active_metrics.get('factor_risk_contribution', 0)
            active_specific = active_metrics.get('specific_risk_contribution', 0)
            active_factor_pct = active_metrics.get('factor_risk_percentage', 0)
            
            # Display metrics
            render_kpi_card("Active Risk", active_risk, "Active total risk", "basis_points")
            render_kpi_card("Active Factor", active_factor, "Active factor contribution", "basis_points")
            render_kpi_card("Active Specific", active_specific, "Active specific contribution", "basis_points")
            render_kpi_card("Active Factor %", active_factor_pct, "Active factor percentage", "percentage")
        else:
            st.info("Active metrics not available for selected node")

def render_risk_composition_chart(data_loader, sidebar_state) -> None:
    """Render stacked bar chart for Factor vs Specific risk composition"""
    
    st.subheader("🏗️ Risk Composition")
    
    # Get metrics for all three lenses
    lenses = ["portfolio", "benchmark", "active"]
    lens_data = {}
    
    for lens in lenses:
        metrics = data_loader.get_core_metrics(lens, sidebar_state.selected_node)
        if metrics:
            lens_data[lens] = {
                'factor': metrics.get('factor_risk_contribution', 0),
                'specific': metrics.get('specific_risk_contribution', 0)
            }
    
    if not lens_data:
        st.info("Risk composition data not available")
        return
    
    # Create stacked bar chart
    fig = go.Figure()
    
    factor_values = [lens_data[lens]['factor'] for lens in lenses if lens in lens_data]
    specific_values = [lens_data[lens]['specific'] for lens in lenses if lens in lens_data] 
    lens_labels = [lens.title() for lens in lenses if lens in lens_data]
    
    # Add factor risk bars
    fig.add_trace(go.Bar(
        x=lens_labels,
        y=factor_values,
        name='Factor Risk',
        marker_color=get_chart_color("factor_risk"),
        text=[f"{v:.0f} bps" for v in factor_values],
        textposition='inside'
    ))
    
    # Add specific risk bars  
    fig.add_trace(go.Bar(
        x=lens_labels,
        y=specific_values,
        name='Specific Risk',
        marker_color=get_chart_color("specific_risk"),
        text=[f"{v:.0f} bps" for v in specific_values],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="Risk Composition: Factor vs Specific",
        barmode='stack',
        xaxis_title="Lens",
        yaxis_title="Risk Contribution (bps)",
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)