import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from components.charts.kpi_cards import render_kpi_cards_portfolio_active, render_risk_composition_chart
from components.charts.risk_charts import render_top_contributors_chart, render_treemap_hierarchy, render_factor_exposures_radar

def render_overview_tab(data_loader, sidebar_state):
    """Render Tab 1 - Overview (snapshot)"""
    
    st.header("Overview - Risk Snapshot")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # KPI cards - Portfolio | Active side-by-side
    render_kpi_cards_portfolio_active(data_loader, sidebar_state)
    
    st.divider()
    
    # Risk composition - stacked bars for Portfolio, Benchmark, Active
    render_risk_composition_chart(data_loader, sidebar_state)
    
    st.divider()
    
    # Two columns for contributors and hierarchy
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Asset Contributors")
        render_top_contributors_chart(
            data_loader, 
            sidebar_state, 
            contrib_type="by_asset",
            title="Top Asset Contributors"
        )
        
        st.subheader("Top Factor Contributors") 
        render_top_contributors_chart(
            data_loader,
            sidebar_state,
            contrib_type="by_factor", 
            title="Top Factor Contributors"
        )
    
    with col2:
        # Hierarchy footprint treemap
        render_treemap_hierarchy(data_loader, sidebar_state)
        
        st.divider()
        
        # Factor posture - radar chart
        render_factor_exposures_radar(data_loader, sidebar_state)
    
    # Additional insights section
    st.divider()
    st.subheader("Summary Insights")
    
    # Get component-specific summary metrics 
    # Try hierarchical data first, fallback to legacy structure
    hierarchical_components = data_loader.get_available_hierarchical_components()
    
    if sidebar_state.selected_node in hierarchical_components:
        # Use new hierarchical data access
        portfolio_metrics = data_loader.get_component_risk_summary(sidebar_state.selected_node, "portfolio")
        active_metrics = data_loader.get_component_risk_summary(sidebar_state.selected_node, "active")
    else:
        # Fallback to legacy data access
        portfolio_metrics = data_loader.get_core_metrics("portfolio", sidebar_state.selected_node)
        active_metrics = data_loader.get_core_metrics("active", sidebar_state.selected_node)
    
    insights = []
    
    if portfolio_metrics:
        total_risk = portfolio_metrics.get('total_risk', 0)
        factor_pct = portfolio_metrics.get('factor_risk_percentage', 0)
        insights.append(f"Portfolio total risk: **{total_risk:.0f} bps**")
        insights.append(f"Factor risk represents **{factor_pct:.1f}%** of total risk")
    
    if active_metrics:
        active_risk = active_metrics.get('total_risk', 0)
        insights.append(f"Active risk: **{active_risk:.0f} bps**")
    
    # Show number of selected factors
    if sidebar_state.selected_factors:
        insights.append(f"Analyzing **{len(sidebar_state.selected_factors)}** selected factors")
    else:
        factor_count = len(data_loader.get_factor_names())
        insights.append(f"All **{factor_count}** factors included in analysis")
    
    # Display insights
    if insights:
        for insight in insights:
            st.markdown(f"• {insight}")
    else:
        st.info("No summary insights available for current selection")