import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color, get_factor_color, get_discrete_color_sequence

def render_top_contributors_chart(
    data_loader,
    sidebar_state,
    contrib_type: str = "by_asset",
    title: str = "Top Contributors"
) -> None:
    """Render horizontal bar chart of top risk contributors"""
    
    lens = sidebar_state.lens
    contributions = data_loader.get_contributions(lens, contrib_type)
    
    if not contributions:
        st.info(f"No {contrib_type.replace('_', ' ')} data available")
        return
    
    # Filter by selected factors if contrib_type is by_factor
    if contrib_type == "by_factor" and sidebar_state.selected_factors:
        contributions = data_loader.filter_data_by_factors(contributions, sidebar_state.selected_factors)
    
    # Sort by absolute value and take top 10
    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    if not sorted_contribs:
        st.info("No contribution data to display")
        return
    
    names, values = zip(*sorted_contribs)
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(names),
        orientation='h',
        marker_color=[get_chart_color("positive") if v >= 0 else get_chart_color("negative") for v in values],
        text=[f"{v:.0f} bps" for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"{title} - {lens.title()} Lens",
        xaxis_title="Contribution (bps)",
        yaxis_title="Component" if contrib_type == "by_asset" else "Factor",
        height=max(300, len(sorted_contribs) * 30 + 100),
        showlegend=False
    )
    
    # Reverse y-axis to show highest contributors at top
    fig.update_yaxes(autorange="reversed")
    
    st.plotly_chart(fig, use_container_width=True)

def render_treemap_hierarchy(data_loader, sidebar_state) -> None:
    """Render treemap of hierarchy footprint for current node"""
    
    st.subheader("🌳 Hierarchy Footprint")
    
    # Get hierarchy and adjacency data
    hierarchy = data_loader.get_hierarchy_info()
    adjacency_list = hierarchy.get('adjacency_list', {})
    
    current_node = sidebar_state.selected_node
    children = adjacency_list.get(current_node, [])
    
    if not children:
        st.info(f"No child components found for {current_node}")
        return
    
    # Get contributions for children
    lens = sidebar_state.lens
    contributions = data_loader.get_contributions(lens, "by_asset")
    
    # Prepare treemap data
    child_data = []
    for child in children:
        contribution = abs(contributions.get(child, 0))
        if contribution > 0:  # Only show components with non-zero contribution
            child_data.append({
                'name': child,
                'contribution': contribution
            })
    
    if not child_data:
        st.info("No child contribution data available")
        return
    
    # Create treemap
    fig = go.Figure(go.Treemap(
        labels=[item['name'] for item in child_data],
        values=[item['contribution'] for item in child_data],
        parents=[""] * len(child_data),  # All are root level
        textinfo="label+value",
        texttemplate="<b>%{label}</b><br>%{value:.0f} bps",
        marker_colorscale=get_discrete_color_sequence(len(child_data)),
        marker_line_width=2
    ))
    
    fig.update_layout(
        title=f"Child Components of {current_node} - {lens.title()} Lens",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_factor_exposures_radar(data_loader, sidebar_state) -> None:
    """Render radar chart for factor exposures"""
    
    st.subheader("🎯 Factor Exposures")
    
    lens = sidebar_state.lens
    exposures = data_loader.get_exposures(lens)
    
    if not exposures:
        st.info("Factor exposure data not available")
        return
    
    # Filter by selected factors if any
    if sidebar_state.selected_factors:
        exposures = data_loader.filter_data_by_factors(exposures, sidebar_state.selected_factors)
    
    factor_names = list(exposures.keys())
    exposure_values = list(exposures.values())
    
    if not factor_names:
        st.info("No factor exposure data to display")
        return
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=exposure_values,
        theta=factor_names,
        fill='toself',
        name=f'{lens.title()} Exposures',
        marker_color=get_chart_color(lens),
        line_color=get_chart_color(lens)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-max(abs(min(exposure_values)), abs(max(exposure_values))) * 1.1,
                       max(abs(min(exposure_values)), abs(max(exposure_values))) * 1.1]
            )),
        title=f"Factor Exposures - {lens.title()} Lens",
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_factor_exposures_bar(data_loader, sidebar_state) -> None:
    """Render bar chart for factor exposures as alternative to radar"""
    
    lens = sidebar_state.lens
    exposures = data_loader.get_exposures(lens)
    
    if not exposures:
        st.info("Factor exposure data not available")
        return
    
    # Filter by selected factors if any
    if sidebar_state.selected_factors:
        exposures = data_loader.filter_data_by_factors(exposures, sidebar_state.selected_factors)
    
    factor_names = list(exposures.keys())
    exposure_values = list(exposures.values())
    
    if not factor_names:
        st.info("No factor exposure data to display")
        return
    
    # Create bar chart
    fig = go.Figure(go.Bar(
        x=factor_names,
        y=exposure_values,
        marker_color=[get_factor_color(name) for name in factor_names],
        text=[f"{v:.3f}" for v in exposure_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Factor Exposures - {lens.title()} Lens",
        xaxis_title="Factors",
        yaxis_title="Exposure",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)