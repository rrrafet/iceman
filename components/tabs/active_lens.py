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
    """Render active KPIs using hierarchical component decomposition data"""
    
    st.subheader("Active Risk KPIs")
    
    # NEW: Get active decomposition data directly from hierarchical schema
    active_decomposition = data_loader.get_component_decomposition(sidebar_state.selected_node, "active")
    
    if not active_decomposition:
        st.info(f"Active risk data not available for {sidebar_state.selected_node}")
        
        # Show diagnostic info about what data is available
        component_summary = data_loader.get_component_risk_summary(sidebar_state.selected_node)
        if component_summary:
            available_lenses = []
            if data_loader.get_component_decomposition(sidebar_state.selected_node, "portfolio"):
                available_lenses.append("portfolio")
            if data_loader.get_component_decomposition(sidebar_state.selected_node, "benchmark"):
                available_lenses.append("benchmark")
            if available_lenses:
                st.info(f"Available lenses for {sidebar_state.selected_node}: {', '.join(available_lenses)}")
        return
    
    # Create 4 columns for KPIs - convert to bps for display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_risk = active_decomposition.get('total_risk', 0) * 10000  # Convert to bps
        st.metric("Active Risk", f"{total_risk:.0f} bps", help="Total active risk vs benchmark")
    
    with col2:
        factor_contrib = active_decomposition.get('factor_risk_contribution', 0) * 10000
        st.metric("Factor Risk", f"{factor_contrib:.0f} bps", help="Active factor risk contribution")
    
    with col3:
        specific_contrib = active_decomposition.get('specific_risk_contribution', 0) * 10000
        st.metric("Specific Risk", f"{specific_contrib:.0f} bps", help="Active specific risk contribution")
    
    with col4:
        # Calculate factor percentage
        total = active_decomposition.get('total_risk', 0)
        factor_pct = (active_decomposition.get('factor_risk_contribution', 0) / total * 100) if total > 0 else 0
        st.metric("Factor %", f"{factor_pct:.1f}%", help="Factor risk as % of total active risk")

def render_tilts_vs_impact_scatter(data_loader, sidebar_state):
    """Render scatter plot using hierarchical active data"""
    
    st.subheader("Tilts vs Impact")
    
    # NEW: Get active weights and contributions using hierarchical schema
    schema_data = data_loader.get_comprehensive_schema_data(sidebar_state.selected_node)
    if not schema_data:
        st.info("No schema data available for tilts analysis")
        return
    
    # Get active weights from schema
    weights_section = schema_data.get('weights', {})
    active_weights = weights_section.get('active_weights', {})
    
    # Get active factor contributions from hierarchical data
    active_contributions = data_loader.get_factor_contributions_from_schema(sidebar_state.selected_node, "active")
    
    if not active_weights and not active_contributions:
        st.info("Active weights and contributions not available")
        
        # Show what's available in the weights section
        if weights_section:
            available_weight_types = list(weights_section.keys())
            st.info(f"Available weight types: {available_weight_types}")
        
        return
    
    # If we have factor contributions but no asset-level active weights,
    # use factor contributions for the analysis
    if active_contributions and not active_weights:
        # Use factor contributions as both tilts and impacts
        common_components = set(active_contributions.keys())
        weights_data = active_contributions  # Factor exposures as "tilts"
        contributions_data = {f: v * 10000 for f, v in active_contributions.items()}  # Convert to bps
    elif active_weights:
        # Use asset-level active weights
        asset_contributions = data_loader.get_contributions("active", "by_asset", sidebar_state.selected_node)
        common_components = set(active_weights.keys()) & set(asset_contributions.keys())
        weights_data = active_weights
        contributions_data = asset_contributions
    else:
        st.info("Insufficient data for tilts vs impact analysis")
        return
    
    if not common_components:
        st.info("No matching components found for analysis")
        return
    
    # Prepare scatter plot data
    weights = []
    contributions = []
    names = []
    
    for component in common_components:
        weights.append(weights_data[component])
        contributions.append(contributions_data[component])
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
                      'Contribution: %{y:.4f} (volatility)<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Active Weights vs Risk Contributions",
        xaxis_title="Active Weight (%)",
        yaxis_title="Active Risk Contribution (volatility)",
        height=400,
        showlegend=False
    )
    
    # Add quadrant lines at zero
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    st.plotly_chart(fig, use_container_width=True)

def render_weight_comparison_bars(data_loader, sidebar_state):
    """Render dual bars comparing portfolio vs benchmark weights using hierarchical data"""
    
    st.subheader("Weight Comparison")
    
    # NEW: Get weights from hierarchical schema data
    schema_data = data_loader.get_comprehensive_schema_data(sidebar_state.selected_node)
    if not schema_data:
        st.info("No schema data available for weight comparison")
        return
    
    weights_section = schema_data.get('weights', {})
    portfolio_weights = weights_section.get('portfolio_weights', {})
    benchmark_weights = weights_section.get('benchmark_weights', {})
    active_weights = weights_section.get('active_weights', {})
    
    # If direct weights not available, try to compute from component decompositions
    if not portfolio_weights and not benchmark_weights:
        # Get child components and their weights/contributions
        children = data_loader.get_component_children_from_schema(sidebar_state.selected_node)
        if children:
            portfolio_weights = {}
            benchmark_weights = {}
            active_weights = {}
            
            for child_id in children:
                # Get portfolio weight/contribution
                portfolio_decomp = data_loader.get_component_decomposition(child_id, "portfolio")
                if portfolio_decomp:
                    portfolio_weights[child_id] = portfolio_decomp.get('total_risk', 0) * 100  # Use risk as proxy for weight
                
                # Get benchmark weight/contribution  
                benchmark_decomp = data_loader.get_component_decomposition(child_id, "benchmark")
                if benchmark_decomp:
                    benchmark_weights[child_id] = benchmark_decomp.get('total_risk', 0) * 100
                
                # Get active weight/contribution
                active_decomp = data_loader.get_component_decomposition(child_id, "active")
                if active_decomp:
                    active_weights[child_id] = active_decomp.get('total_risk', 0) * 10000  # Convert to bps
    
    if not portfolio_weights and not benchmark_weights:
        st.info("Portfolio and benchmark weights not available")
        return
    
    # Find common components and take top 10 by absolute active weight/contribution
    common_components = set(portfolio_weights.keys()) & set(benchmark_weights.keys())
    
    if active_weights:
        # Sort by absolute active weight/contribution
        sorted_components = sorted(
            [c for c in common_components if c in active_weights],
            key=lambda x: abs(active_weights[x]),
            reverse=True
        )[:10]
    else:
        sorted_components = list(common_components)[:10]
    
    if not sorted_components:
        st.info("No matching components found for weight comparison")
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
    
    st.subheader("Active Factor Story")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Factor Contributions**")
        
        # NEW: Get active factor contributions from hierarchical data
        active_factor_contribs = data_loader.get_factor_contributions_from_schema(
            sidebar_state.selected_node, "active"
        )
        
        if active_factor_contribs:
            # Filter by selected factors if any
            if sidebar_state.selected_factors:
                active_factor_contribs = {
                    f: v for f, v in active_factor_contribs.items() 
                    if f in sidebar_state.selected_factors
                }
            
            # Convert to bps and sort by absolute value
            active_factor_contribs_bps = {f: v * 10000 for f, v in active_factor_contribs.items()}
            sorted_contribs = sorted(
                active_factor_contribs_bps.items(), 
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
        
        # NEW: Get active exposures from hierarchical data
        active_exposures = data_loader.get_exposures("active", sidebar_state.selected_node)
        
        if active_exposures:
            # Filter by selected factors if any
            if sidebar_state.selected_factors:
                active_exposures = {
                    f: v for f, v in active_exposures.items() 
                    if f in sidebar_state.selected_factors
                }
            
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
    """Render matrix data using hierarchical schema data"""
    
    st.subheader("Matrix Analysis")
    
    # NEW: Get matrix data from hierarchical schema
    schema_data = data_loader.get_comprehensive_schema_data(sidebar_state.selected_node)
    if not schema_data:
        st.info("No matrix data available")
        return
    
    matrices_section = schema_data.get('matrices', {})
    arrays_section = schema_data.get('arrays', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Weighted Betas (Active)**")
        active_betas = matrices_section.get('weighted_betas', {})
        
        if active_betas:
            # Show top factor betas
            sorted_betas = sorted(active_betas.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            if sorted_betas:
                factors, betas = zip(*sorted_betas)
                
                fig = go.Figure(go.Bar(
                    x=list(betas),
                    y=list(factors),
                    orientation='h',
                    marker_color=[get_chart_color("positive") if v >= 0 else get_chart_color("negative") for v in betas],
                    text=[f"{v:.3f}" for v in betas],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Active Factor Betas",
                    xaxis_title="Beta",
                    yaxis_title="Factors",
                    height=250,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No beta data to display")
        else:
            st.info("Active beta data not available")
    
    with col2:
        st.markdown("**Factor Risk Contributions**")
        factor_risk_contribs = matrices_section.get('factor_risk_contributions', {})
        
        if factor_risk_contribs:
            # Show top factor risk contributions
            sorted_contribs = sorted(factor_risk_contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            if sorted_contribs:
                factors, contribs = zip(*sorted_contribs)
                contribs_bps = [c * 10000 for c in contribs]  # Convert to bps
                
                fig = go.Figure(go.Bar(
                    x=contribs_bps,
                    y=list(factors),
                    orientation='h',
                    marker_color=[get_chart_color("positive") if v >= 0 else get_chart_color("negative") for v in contribs_bps],
                    text=[f"{v:.0f} bps" for v in contribs_bps],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Factor Risk Contributions",
                    xaxis_title="Contribution (bps)",
                    yaxis_title="Factors",
                    height=250,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No factor risk contribution data to display")
        else:
            st.info("Factor risk contribution data not available")