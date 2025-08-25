import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color, get_factor_color

def render_active_lens_tab(data_loader, sidebar_state):
    """Render Tab 2 - Active lens (why we're different)"""
    
    st.header("Active Lens - Portfolio vs Benchmark Differences")
    st.markdown(f"**Current Node:** {sidebar_state.selected_node}")
    
    # Active KPIs
    render_active_kpis(data_loader, sidebar_state)
    
    st.divider()
    
    # Two columns for tilts analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Tilts vs impact scatter plot
        render_tilts_vs_impact_scatter(data_loader, sidebar_state)
    
    with col2:
        # Weight comparison bars
        render_weight_comparison_bars(data_loader, sidebar_state)
    
    st.divider()
    
    # Active factor story
    render_active_factor_story(data_loader, sidebar_state)
    
    st.divider()
    
    # Matrices section (if available)
    render_active_matrices(data_loader, sidebar_state)

def render_active_kpis(data_loader, sidebar_state):
    """Render active-specific KPIs"""
    
    st.subheader("🎯 Active Risk KPIs")
    
    active_metrics = data_loader.get_core_metrics("active", sidebar_state.selected_node)
    
    if not active_metrics:
        st.info("Active metrics not available for selected node")
        return
    
    # Create 4 columns for KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_risk = active_metrics.get('total_risk', 0)
        st.metric("Active Risk", f"{total_risk:.0f} bps", help="Total active risk vs benchmark")
    
    with col2:
        factor_contrib = active_metrics.get('factor_risk_contribution', 0)
        st.metric("Factor Risk", f"{factor_contrib:.0f} bps", help="Active factor risk contribution")
    
    with col3:
        specific_contrib = active_metrics.get('specific_risk_contribution', 0)
        st.metric("Specific Risk", f"{specific_contrib:.0f} bps", help="Active specific risk contribution")
    
    with col4:
        factor_pct = active_metrics.get('factor_risk_percentage', 0)
        st.metric("Factor %", f"{factor_pct:.1f}%", help="Factor risk as % of total active risk")

def render_tilts_vs_impact_scatter(data_loader, sidebar_state):
    """Render scatter plot of active weights vs active contributions"""
    
    st.subheader("📊 Tilts vs Impact")
    
    # Get active weights and contributions
    active_weights = data_loader.get_weights("active_weights")
    active_contributions = data_loader.get_contributions("active", "by_asset")
    
    if not active_weights or not active_contributions:
        st.info("Active weights or contributions not available")
        return
    
    # Find common components
    common_components = set(active_weights.keys()) & set(active_contributions.keys())
    
    if not common_components:
        st.info("No matching components found")
        return
    
    # Prepare scatter plot data
    weights = []
    contributions = []
    names = []
    
    for component in common_components:
        weights.append(active_weights[component])
        contributions.append(active_contributions[component])
        names.append(component)
    
    # Get component metadata for hover info
    hierarchy = data_loader.get_hierarchy_info()
    component_metadata = hierarchy.get('component_metadata', {})
    
    hover_text = []
    for component in names:
        metadata = component_metadata.get(component, {})
        comp_type = metadata.get('type', 'N/A')
        level = metadata.get('level', 'N/A')
        hover_text.append(f"{component}<br>Type: {comp_type}<br>Level: {level}")
    
    # Create scatter plot
    fig = go.Figure(go.Scatter(
        x=weights,
        y=contributions,
        mode='markers',
        marker=dict(
            size=10,
            color=get_chart_color("active"),
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=names,
        hovertext=hover_text,
        hovertemplate='<b>%{hovertext}</b><br>' +
                      'Active Weight: %{x:.2f}%<br>' +
                      'Contribution: %{y:.0f} bps<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Active Weights vs Risk Contributions",
        xaxis_title="Active Weight (%)",
        yaxis_title="Active Risk Contribution (bps)",
        height=400,
        showlegend=False
    )
    
    # Add quadrant lines at zero
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    st.plotly_chart(fig, use_container_width=True)

def render_weight_comparison_bars(data_loader, sidebar_state):
    """Render dual bars comparing portfolio vs benchmark weights"""
    
    st.subheader("⚖️ Weight Comparison")
    
    # Get weights
    portfolio_weights = data_loader.get_weights("portfolio_weights")
    benchmark_weights = data_loader.get_weights("benchmark_weights")
    active_weights = data_loader.get_weights("active_weights")
    
    if not portfolio_weights or not benchmark_weights:
        st.info("Portfolio or benchmark weights not available")
        return
    
    # Find common components and take top 10 by absolute active weight
    common_components = set(portfolio_weights.keys()) & set(benchmark_weights.keys())
    
    if active_weights:
        # Sort by absolute active weight
        sorted_components = sorted(
            [c for c in common_components if c in active_weights],
            key=lambda x: abs(active_weights[x]),
            reverse=True
        )[:10]
    else:
        sorted_components = list(common_components)[:10]
    
    if not sorted_components:
        st.info("No matching components found")
        return
    
    # Prepare data
    port_vals = [portfolio_weights.get(c, 0) for c in sorted_components]
    bench_vals = [benchmark_weights.get(c, 0) for c in sorted_components]
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Portfolio',
        x=sorted_components,
        y=port_vals,
        marker_color=get_chart_color("portfolio"),
        text=[f"{v:.1f}%" for v in port_vals],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Benchmark',
        x=sorted_components,
        y=bench_vals,
        marker_color=get_chart_color("benchmark"),
        text=[f"{v:.1f}%" for v in bench_vals],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Portfolio vs Benchmark Weights",
        xaxis_title="Components",
        yaxis_title="Weight (%)",
        barmode='group',
        height=400,
        showlegend=True
    )
    
    # Rotate x-axis labels for readability
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)

def render_active_factor_story(data_loader, sidebar_state):
    """Render active factor contributions and exposures"""
    
    st.subheader("🔍 Active Factor Story")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Factor Contributions**")
        
        active_factor_contribs = data_loader.get_contributions("active", "by_factor")
        
        if active_factor_contribs:
            # Filter by selected factors if any
            if sidebar_state.selected_factors:
                active_factor_contribs = data_loader.filter_data_by_factors(
                    active_factor_contribs, 
                    sidebar_state.selected_factors
                )
            
            # Sort by absolute value
            sorted_contribs = sorted(
                active_factor_contribs.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            if sorted_contribs:
                factors, contributions = zip(*sorted_contribs)
                
                fig = go.Figure(go.Bar(
                    x=list(contributions),
                    y=list(factors),
                    orientation='h',
                    marker_color=[get_chart_color("positive") if v >= 0 else get_chart_color("negative") for v in contributions],
                    text=[f"{v:.0f} bps" for v in contributions],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Active Factor Contributions",
                    xaxis_title="Contribution (bps)",
                    yaxis_title="Factors",
                    height=300,
                    showlegend=False
                )
                
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No factor contribution data")
        else:
            st.info("Active factor contributions not available")
    
    with col2:
        st.markdown("**Factor Exposures (Tilts)**")
        
        active_exposures = data_loader.get_exposures("active")
        
        if active_exposures:
            # Filter by selected factors if any
            if sidebar_state.selected_factors:
                active_exposures = data_loader.filter_data_by_factors(
                    active_exposures,
                    sidebar_state.selected_factors
                )
            
            factors = list(active_exposures.keys())
            exposures = list(active_exposures.values())
            
            if factors:
                fig = go.Figure(go.Bar(
                    x=factors,
                    y=exposures,
                    marker_color=[get_factor_color(f) for f in factors],
                    text=[f"{v:.3f}" for v in exposures],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Active Factor Exposures",
                    xaxis_title="Factors",
                    yaxis_title="Exposure",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No exposure data to display")
        else:
            st.info("Active factor exposures not available")

def render_active_matrices(data_loader, sidebar_state):
    """Render matrix data if available"""
    
    st.subheader("📋 Matrix Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Beta Matrix**")
        beta_matrix = data_loader.get_matrices("beta_matrix")
        if beta_matrix:
            st.info("Beta matrix visualization coming soon")
        else:
            st.info("Beta matrix not available - will populate when computed")
    
    with col2:
        st.markdown("**Factor Risk Contributions**")
        factor_risk_matrix = data_loader.get_matrices("factor_risk_contributions")
        if factor_risk_matrix:
            st.info("Factor risk matrix visualization coming soon")
        else:
            st.info("Factor risk matrix not available - will populate when computed")