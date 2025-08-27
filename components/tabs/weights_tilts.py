import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color

def render_weights_tilts_tab(data_loader, sidebar_state):
    """Render Tab 7 - Weights & tilts"""
    
    st.header("Weights & Tilts - Position Analysis")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # Weight comparison section
    render_weight_comparison(data_loader, sidebar_state)
    
    st.divider()
    
    # Concentration analysis
    render_concentration_analysis(data_loader, sidebar_state)

def render_weight_comparison(data_loader, sidebar_state):
    """Render weight comparison charts"""
    
    st.subheader("Weight Comparison")
    
    # Get weights data
    portfolio_weights = data_loader.get_weights("portfolio_weights")
    benchmark_weights = data_loader.get_weights("benchmark_weights")
    active_weights = data_loader.get_weights("active_weights")
    
    if not portfolio_weights and not benchmark_weights:
        st.info("No weight data available")
        return
    
    # Two columns for different views
    col1, col2 = st.columns(2)
    
    with col1:
        render_grouped_weight_bars(portfolio_weights, benchmark_weights, sidebar_state)
    
    with col2:
        render_active_weight_bars(active_weights, sidebar_state)

def render_grouped_weight_bars(portfolio_weights, benchmark_weights, sidebar_state):
    """Render grouped bars comparing portfolio vs benchmark weights"""
    
    st.markdown("**Portfolio vs Benchmark Weights**")
    
    if not portfolio_weights or not benchmark_weights:
        st.info("Need both portfolio and benchmark weights for comparison")
        return
    
    # Find common components and get top positions
    common_components = set(portfolio_weights.keys()) & set(benchmark_weights.keys())
    
    # Sort by portfolio weight magnitude
    sorted_components = sorted(
        common_components,
        key=lambda x: abs(portfolio_weights.get(x, 0)),
        reverse=True
    )[:20]  # Top 20 positions
    
    if not sorted_components:
        st.info("No common components found")
        return
    
    # Prepare data
    components = []
    port_vals = []
    bench_vals = []
    
    for component in sorted_components:
        components.append(component)
        port_vals.append(portfolio_weights.get(component, 0))
        bench_vals.append(benchmark_weights.get(component, 0))
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Portfolio',
        x=components,
        y=port_vals,
        marker_color=get_chart_color("portfolio"),
        text=[f"{v:.1f}%" for v in port_vals],
        textposition='outside',
        hovertemplate='Portfolio<br>%{x}: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Benchmark',
        x=components,
        y=bench_vals,
        marker_color=get_chart_color("benchmark"),
        text=[f"{v:.1f}%" for v in bench_vals],
        textposition='outside',
        hovertemplate='Benchmark<br>%{x}: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Top Holdings: Portfolio vs Benchmark",
        xaxis_title="Components",
        yaxis_title="Weight (%)",
        barmode='group',
        height=400,
        showlegend=True
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)

def render_active_weight_bars(active_weights, sidebar_state):
    """Render active weights with highlighting of extremes"""
    
    st.markdown("**Active Weights (Tilts)**")
    
    if not active_weights:
        st.info("No active weight data available")
        return
    
    # Sort by absolute active weight
    sorted_weights = sorted(
        active_weights.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:20]  # Top 20 active positions
    
    if not sorted_weights:
        st.info("No active weight data to display")
        return
    
    components, weights = zip(*sorted_weights)
    
    # Color bars based on sign (green for positive, red for negative)
    colors = []
    for weight in weights:
        if weight > 0:
            colors.append(get_chart_color("positive"))
        elif weight < 0:
            colors.append(get_chart_color("negative"))
        else:
            colors.append(get_chart_color("neutral"))
    
    # Create bar chart
    fig = go.Figure(go.Bar(
        x=list(components),
        y=list(weights),
        marker_color=colors,
        text=[f"{v:.1f}%" for v in weights],
        textposition='outside',
        hovertemplate='%{x}<br>Active Weight: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Active Weights (Top Tilts)",
        xaxis_title="Components",
        yaxis_title="Active Weight (%)",
        height=400,
        showlegend=False
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show weight statistics
    render_weight_statistics(weights)

def render_weight_statistics(active_weights):
    """Render statistics about active weights"""
    
    st.markdown("**Active Weight Statistics**")
    
    weights = list(active_weights)
    abs_weights = [abs(w) for w in weights]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_positive = max([w for w in weights if w > 0], default=0)
        st.metric("Max Overweight", f"{max_positive:.2f}%")
    
    with col2:
        min_negative = min([w for w in weights if w < 0], default=0)
        st.metric("Max Underweight", f"{min_negative:.2f}%")
    
    with col3:
        avg_abs_weight = sum(abs_weights) / len(abs_weights) if abs_weights else 0
        st.metric("Avg Abs Weight", f"{avg_abs_weight:.2f}%")
    
    with col4:
        negative_count = len([w for w in weights if w < 0])
        st.metric("Negative Positions", negative_count)

def render_concentration_analysis(data_loader, sidebar_state):
    """Render concentration analysis including Pareto chart"""
    
    st.subheader("Concentration Analysis")
    
    # Get active weights and contributions for Pareto analysis
    active_weights = data_loader.get_weights("active_weights")
    contributions = data_loader.get_contributions(sidebar_state.lens, "by_asset")
    
    if not active_weights or not contributions:
        st.info("Need both active weights and contributions for concentration analysis")
        return
    
    # Two columns for different concentration views
    col1, col2 = st.columns(2)
    
    with col1:
        render_pareto_chart(active_weights, contributions, sidebar_state)
    
    with col2:
        render_concentration_metrics(active_weights, contributions)

def render_pareto_chart(active_weights, contributions, sidebar_state):
    """Render Pareto chart of cumulative contribution vs cumulative active weight"""
    
    st.markdown("**Pareto Analysis: Risk vs Active Weight**")
    
    # Find common components
    common_components = set(active_weights.keys()) & set(contributions.keys())
    
    if not common_components:
        st.info("No common components for Pareto analysis")
        return
    
    # Prepare data for Pareto chart
    data_points = []
    for component in common_components:
        weight = abs(active_weights.get(component, 0))
        contrib = abs(contributions.get(component, 0))
        data_points.append({
            'component': component,
            'abs_weight': weight,
            'abs_contrib': contrib
        })
    
    # Sort by absolute contribution (descending)
    data_points.sort(key=lambda x: x['abs_contrib'], reverse=True)
    
    # Calculate cumulative values
    total_weight = sum(point['abs_weight'] for point in data_points)
    total_contrib = sum(point['abs_contrib'] for point in data_points)
    
    cumulative_weight = 0
    cumulative_contrib = 0
    
    components = []
    cum_weight_pct = []
    cum_contrib_pct = []
    
    for point in data_points:
        cumulative_weight += point['abs_weight']
        cumulative_contrib += point['abs_contrib']
        
        components.append(point['component'])
        cum_weight_pct.append((cumulative_weight / total_weight) * 100 if total_weight > 0 else 0)
        cum_contrib_pct.append((cumulative_contrib / total_contrib) * 100 if total_contrib > 0 else 0)
    
    # Create Pareto chart
    fig = go.Figure()
    
    # Scatter plot for Pareto curve
    fig.add_trace(go.Scatter(
        x=cum_weight_pct,
        y=cum_contrib_pct,
        mode='lines+markers',
        name='Pareto Curve',
        line=dict(color=get_chart_color("active"), width=3),
        marker=dict(size=8),
        hovertemplate='Cumulative Weight: %{x:.1f}%<br>Cumulative Risk: %{y:.1f}%<extra></extra>'
    ))
    
    # Add diagonal reference line (perfect distribution)
    fig.add_trace(go.Scatter(
        x=[0, 100],
        y=[0, 100],
        mode='lines',
        name='Perfect Distribution',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=False
    ))
    
    fig.update_layout(
        title="Cumulative Risk vs Cumulative Active Weight",
        xaxis_title="Cumulative Active Weight (%)",
        yaxis_title="Cumulative Risk Contribution (%)",
        height=400,
        showlegend=True
    )
    
    # Equal axis scaling
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show concentration insights
    render_pareto_insights(cum_weight_pct, cum_contrib_pct)

def render_pareto_insights(cum_weight_pct, cum_contrib_pct):
    """Render insights from Pareto analysis"""
    
    if not cum_weight_pct or not cum_contrib_pct:
        return
    
    st.markdown("**Concentration Insights**")
    
    # Find key percentiles
    percentiles = [20, 50, 80]
    insights = []
    
    for pct in percentiles:
        # Find closest index to weight percentile
        target_weight = pct
        closest_idx = min(range(len(cum_weight_pct)), 
                         key=lambda i: abs(cum_weight_pct[i] - target_weight))
        
        if closest_idx < len(cum_contrib_pct):
            contrib_at_pct = cum_contrib_pct[closest_idx]
            insights.append(f"Top {pct}% of positions drive {contrib_at_pct:.1f}% of risk")
    
    for insight in insights:
        st.markdown(f"• {insight}")

def render_concentration_metrics(active_weights, contributions):
    """Render concentration metrics and statistics"""
    
    st.markdown("**Concentration Metrics**")
    
    # Calculate concentration measures
    abs_weights = [abs(w) for w in active_weights.values()]
    abs_contribs = [abs(c) for c in contributions.values()]
    
    # Herfindahl-Hirschman Index (HHI) for weights
    total_abs_weight = sum(abs_weights) if abs_weights else 1
    normalized_weights = [w / total_abs_weight for w in abs_weights] if total_abs_weight > 0 else []
    hhi_weights = sum(w**2 for w in normalized_weights) * 10000 if normalized_weights else 0
    
    # HHI for contributions
    total_abs_contrib = sum(abs_contribs) if abs_contribs else 1
    normalized_contribs = [c / total_abs_contrib for c in abs_contribs] if total_abs_contrib > 0 else []
    hhi_contribs = sum(c**2 for c in normalized_contribs) * 10000 if normalized_contribs else 0
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Weight HHI", f"{hhi_weights:.0f}", 
                 help="Higher values indicate more concentrated positions")
        
        # Top N concentration
        top_5_weight_pct = sum(sorted(abs_weights, reverse=True)[:5]) / sum(abs_weights) * 100 if abs_weights else 0
        st.metric("Top 5 Weight %", f"{top_5_weight_pct:.1f}%")
    
    with col2:
        st.metric("Risk HHI", f"{hhi_contribs:.0f}",
                 help="Higher values indicate more concentrated risk")
        
        # Top N risk concentration
        top_5_risk_pct = sum(sorted(abs_contribs, reverse=True)[:5]) / sum(abs_contribs) * 100 if abs_contribs else 0
        st.metric("Top 5 Risk %", f"{top_5_risk_pct:.1f}%")
    
    # Concentration interpretation
    st.markdown("**Interpretation:**")
    
    if hhi_weights > 2000:
        st.warning("High position concentration - consider diversification")
    elif hhi_weights > 1000:
        st.info("Moderate position concentration")
    else:
        st.success("Well-diversified positions")
    
    if hhi_contribs > 2000:
        st.warning("High risk concentration - few positions drive most risk")
    elif hhi_contribs > 1000:
        st.info("Moderate risk concentration")
    else:
        st.success("Well-diversified risk profile")
    
    # Show top concentrations
    st.markdown("**Top Risk Concentrators:**")
    
    # Combine weights and contributions
    common_components = set(active_weights.keys()) & set(contributions.keys())
    component_data = []
    
    for component in common_components:
        component_data.append({
            'component': component,
            'abs_weight': abs(active_weights[component]),
            'abs_contrib': abs(contributions[component]),
            'efficiency': abs(contributions[component]) / abs(active_weights[component]) if abs(active_weights[component]) > 0.01 else 0
        })
    
    # Sort by risk contribution
    component_data.sort(key=lambda x: x['abs_contrib'], reverse=True)
    
    # Show top 5
    for i, item in enumerate(component_data[:5], 1):
        st.markdown(f"{i}. **{item['component']}**: {item['abs_contrib']:.1f} bps risk from {item['abs_weight']:.2f}% weight (efficiency: {item['efficiency']:.1f})")