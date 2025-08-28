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
    contrib_type: str = "by_factor",
    title: str = "Top Contributors"
) -> None:
    """SIMPLIFIED: Render horizontal bar chart using direct schema delegation"""
    
    # Direct schema access - single source of truth
    contributions = data_loader.get_contributions(sidebar_state.lens, contrib_type)
    
    if not contributions:
        st.info(f"No {contrib_type.replace('_', ' ')} data available: check schema.get_ui_contributions({sidebar_state.selected_node}, '{sidebar_state.lens}', '{contrib_type}')")
        return
    
    # Sort by absolute value and take top 10
    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    if not sorted_contribs:
        st.info("No contribution data to display")
        return
    
    names, values = zip(*sorted_contribs)
    
    # Create horizontal bar chart - values in volatility units
    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(names),
        orientation='h',
        marker_color=[get_chart_color("positive") if v >= 0 else get_chart_color("negative") for v in values],
        text=[f"{v:.4f}" for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"{title} - {sidebar_state.lens.title()} Lens",
        xaxis_title="Contribution (volatility)",
        yaxis_title="Component" if contrib_type == "by_asset" else "Factor",
        height=max(300, len(sorted_contribs) * 30 + 100),
        showlegend=False
    )
    
    # Reverse y-axis to show highest contributors at top
    fig.update_yaxes(autorange="reversed")
    
    st.plotly_chart(fig, use_container_width=True)

def render_treemap_hierarchy(data_loader, sidebar_state) -> None:
    """SIMPLIFIED: Render treemap using direct schema delegation"""
    
    st.subheader("Hierarchy Footprint")
    
    # Direct schema access - single source of truth
    contributions = data_loader.get_contributions(sidebar_state.lens, "by_component")
    
    if not contributions:
        st.info(f"No component contribution data available: check schema.get_ui_contributions({sidebar_state.selected_node}, '{sidebar_state.lens}', 'by_component')")
        return
    
    # Prepare treemap data
    child_data = []
    for name, contribution in contributions.items():
        abs_contrib = abs(contribution)
        if abs_contrib > 0:  # Only show components with non-zero contribution
            child_data.append({
                'name': name,
                'contribution': abs_contrib
            })
    
    if not child_data:
        st.info("No child contribution data available")
        return
    
    # Sort by contribution size
    child_data.sort(key=lambda x: x['contribution'], reverse=True)
    
    # Create treemap - values in volatility units
    fig = go.Figure(go.Treemap(
        labels=[item['name'] for item in child_data],
        values=[item['contribution'] for item in child_data],
        parents=[""] * len(child_data),  # All are root level
        textinfo="label+value",
        texttemplate="<b>%{label}</b><br>%{value:.4f}",
        marker_colorscale=get_discrete_color_sequence(len(child_data)),
        marker_line_width=2
    ))
    
    fig.update_layout(
        title=f"Component Contributions - {sidebar_state.lens.title()} Lens",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_factor_exposures_radar(data_loader, sidebar_state) -> None:
    """SIMPLIFIED: Render radar chart using direct schema delegation"""
    
    st.subheader("Factor Exposures")
    
    # Direct schema access - single source of truth
    exposures = data_loader.get_exposures(sidebar_state.lens)
    
    if not exposures:
        st.info(f"Factor exposure data not available: check schema.get_ui_exposures({sidebar_state.selected_node}, '{sidebar_state.lens}')")
        return
    
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
        name=f'{sidebar_state.lens.title()} Exposures',
        marker_color=get_chart_color(sidebar_state.lens),
        line_color=get_chart_color(sidebar_state.lens)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-max(abs(min(exposure_values)), abs(max(exposure_values))) * 1.1,
                       max(abs(min(exposure_values)), abs(max(exposure_values))) * 1.1]
            )),
        title=f"Factor Exposures - {sidebar_state.lens.title()} Lens",
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_factor_exposures_bar(data_loader, sidebar_state) -> None:
    """SIMPLIFIED: Render bar chart using direct schema delegation"""
    
    # Direct schema access - single source of truth
    exposures = data_loader.get_exposures(sidebar_state.lens)
    
    if not exposures:
        st.info(f"Factor exposure data not available: check schema.get_ui_exposures({sidebar_state.selected_node}, '{sidebar_state.lens}')")
        return
    
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
        title=f"Factor Exposures - {sidebar_state.lens.title()} Lens",
        xaxis_title="Factors",
        yaxis_title="Exposure",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)