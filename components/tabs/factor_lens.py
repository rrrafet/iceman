import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color, get_factor_color

def render_factor_lens_tab(data_loader, sidebar_state):
    """Render Tab 5 - Factor lens (cross-section + time)"""
    
    st.header("Factor Lens - Factor Analysis")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # Factor selection info
    render_factor_selection_info(sidebar_state, data_loader)
    
    st.divider()
    
    # Snapshot section
    render_factor_snapshot(data_loader, sidebar_state)
    
    st.divider()
    
    # Over time section
    render_factor_over_time(data_loader, sidebar_state)
    
    st.divider()
    
    # Matrix section (when available)
    render_factor_matrices(data_loader, sidebar_state)

def render_factor_selection_info(sidebar_state, data_loader):
    """Display current factor selection and allow filtering"""
    
    st.subheader("Factor Selection")
    
    all_factors = data_loader.get_factor_names()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if sidebar_state.selected_factors:
            st.metric("Selected Factors", len(sidebar_state.selected_factors))
            st.markdown("**Active Factors:**")
            for factor in sidebar_state.selected_factors:
                st.markdown(f"• {factor}")
        else:
            st.metric("Selected Factors", "All")
            st.markdown("**All factors included in analysis**")
    
    with col2:
        st.metric("Total Available", len(all_factors))
        
        # Factor filter status
        if sidebar_state.selected_factors:
            filter_pct = len(sidebar_state.selected_factors) / len(all_factors) * 100
            st.metric("Filter Coverage", f"{filter_pct:.1f}%")

def render_factor_snapshot(data_loader, sidebar_state):
    """Render snapshot factor analysis"""
    
    st.subheader("Factor Snapshot")
    
    current_lens = sidebar_state.lens
    
    # Two columns for contributions and exposures
    col1, col2 = st.columns(2)
    
    with col1:
        # Factor contributions bar chart
        render_factor_contributions_chart(data_loader, sidebar_state, current_lens)
    
    with col2:
        # Factor exposures radar/bar chart
        render_factor_exposures_chart(data_loader, sidebar_state, current_lens)

def render_factor_contributions_chart(data_loader, sidebar_state, lens):
    """Render factor contributions bar chart (clickable)"""
    
    st.markdown("**Factor Contributions**")
    
    # NEW: Get factor contributions using hierarchical schema
    factor_contribs = data_loader.get_factor_contributions_from_schema(
        sidebar_state.selected_node, lens
    )
    
    if not factor_contribs:
        st.info(f"No factor contributions available for {lens} lens")
        return
    
    # Filter by selected factors if any
    if sidebar_state.selected_factors:
        factor_contribs = {
            f: v for f, v in factor_contribs.items() 
            if f in sidebar_state.selected_factors
        }
    
    if not factor_contribs:
        st.info("No factor contributions to display with current filter")
        return
    
    # Convert to bps for display
    factor_contribs_bps = {f: v * 10000 for f, v in factor_contribs.items()}
    
    # Sort by absolute value
    sorted_contribs = sorted(
        factor_contribs_bps.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    factors, contributions = zip(*sorted_contribs)
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=list(contributions),
        y=list(factors),
        orientation='h',
        marker_color=[
            get_chart_color("positive") if v >= 0 else get_chart_color("negative") 
            for v in contributions
        ],
        text=[f"{v:.0f} bps" for v in contributions],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' +
                      'Contribution: %{x:.0f} bps<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Factor Contributions - {lens.title()}",
        xaxis_title="Contribution (bps)",
        yaxis_title="Factors",
        height=max(300, len(factors) * 20 + 100),
        showlegend=False
    )
    
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    
    # Click instruction
    st.caption("Click on a factor in the chart to filter other visualizations (feature coming soon)")

def render_factor_exposures_chart(data_loader, sidebar_state, lens):
    """Render factor exposures as bar or radar chart"""
    
    st.markdown("**Factor Exposures**")
    
    # NEW: Get factor exposures using hierarchical schema
    exposures = data_loader.get_exposures(lens, sidebar_state.selected_node)
    
    if not exposures:
        st.info(f"No factor exposures available for {lens} lens")
        return
    
    # Filter by selected factors if any
    if sidebar_state.selected_factors:
        exposures = {
            f: v for f, v in exposures.items() 
            if f in sidebar_state.selected_factors
        }
    
    if not exposures:
        st.info("No factor exposures to display with current filter")
        return
    
    # Chart type selector
    chart_type = st.radio(
        "Chart Type",
        ["Bar Chart", "Radar Chart"],
        horizontal=True,
        key="exposure_chart_type"
    )
    
    factors = list(exposures.keys())
    values = list(exposures.values())
    
    if chart_type == "Bar Chart":
        # Create bar chart
        fig = go.Figure(go.Bar(
            x=factors,
            y=values,
            marker_color=[get_factor_color(f) for f in factors],
            text=[f"{v:.3f}" for v in values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                          'Exposure: %{y:.3f}<br>' +
                          '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Factor Exposures - {lens.title()}",
            xaxis_title="Factors",
            yaxis_title="Exposure",
            height=300,
            showlegend=False
        )
        
        # Rotate x-axis labels if many factors
        if len(factors) > 6:
            fig.update_xaxes(tickangle=45)
    
    else:
        # Create radar chart
        fig = go.Figure(go.Scatterpolar(
            r=values,
            theta=factors,
            fill='toself',
            name=f"{lens.title()} Exposures",
            line_color=get_chart_color(lens)
        ))
        
        fig.update_layout(
            title=f"Factor Exposures - {lens.title()}",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[min(values + [0]) * 1.1, max(values + [0]) * 1.1]
                )
            ),
            showlegend=False,
            height=400
        )
    
    st.plotly_chart(fig, use_container_width=True)

def render_factor_over_time(data_loader, sidebar_state):
    """Render factor analysis over time"""
    
    st.subheader("Factors Over Time")
    
    # Factor selector for time series
    all_factors = data_loader.get_factor_names()
    
    # Pre-select factors based on sidebar filter or top contributors
    default_factors = []
    if sidebar_state.selected_factors:
        default_factors = sidebar_state.selected_factors[:3]  # Limit to top 3
    else:
        # Get top contributing factors using hierarchical schema
        factor_contribs = data_loader.get_factor_contributions_from_schema(
            sidebar_state.selected_node, sidebar_state.lens
        )
        if factor_contribs:
            sorted_factors = sorted(
                factor_contribs.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            default_factors = [f[0] for f in sorted_factors[:3]]
    
    selected_time_factors = st.multiselect(
        "Select factors for time series analysis",
        options=all_factors,
        default=default_factors,
        max_selections=5,
        key="factor_time_series_selector"
    )
    
    if not selected_time_factors:
        st.info("Select factors to view time series")
        return
    
    # Create time series chart
    fig = go.Figure()
    
    for i, factor in enumerate(selected_time_factors):
        # Get factor returns data (placeholder - would come from time_series.factor_returns)
        factor_data = data_loader.get_time_series_data('factor_returns', factor)
        
        if factor_data:
            # Apply date range filter
            start_idx, end_idx = sidebar_state.date_range
            if start_idx > 0 or end_idx < len(factor_data) - 1:
                factor_data = factor_data[start_idx:end_idx + 1]
            
            periods = list(range(len(factor_data)))
            
            fig.add_trace(go.Scatter(
                x=periods,
                y=factor_data,
                mode='lines',
                name=factor,
                line=dict(color=get_factor_color(factor), width=2),
                hovertemplate=f'{factor}<br>Period: %{{x}}<br>Return: %{{y:.4f}}<extra></extra>'
            ))
    
    if fig.data:
        fig.update_layout(
            title="Factor Returns Over Time",
            xaxis_title="Period",
            yaxis_title="Factor Return",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add factor correlation analysis
        if len(selected_time_factors) > 1:
            render_factor_correlation_matrix(selected_time_factors, data_loader, sidebar_state)
    else:
        st.info("No factor time series data available")

def render_factor_correlation_matrix(factors, data_loader, sidebar_state):
    """Render correlation matrix for selected factors"""
    
    st.markdown("**Factor Correlation Matrix**")
    
    # Get correlation data (simplified computation for now)
    correlation_matrix = []
    factor_names = []
    
    for factor1 in factors:
        factor1_data = data_loader.get_time_series_data('factor_returns', factor1)
        if not factor1_data:
            continue
            
        factor_names.append(factor1)
        correlations = []
        
        for factor2 in factors:
            factor2_data = data_loader.get_time_series_data('factor_returns', factor2)
            if not factor2_data:
                correlations.append(0.0)
                continue
            
            # Simple correlation calculation (placeholder)
            if factor1 == factor2:
                correlations.append(1.0)
            else:
                # Would implement proper correlation calculation here
                correlations.append(0.5)  # Placeholder
        
        correlation_matrix.append(correlations)
    
    if correlation_matrix:
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=factor_names,
            y=factor_names,
            colorscale='RdBu',
            zmid=0,
            text=[[f"{val:.2f}" for val in row] for row in correlation_matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Factor Correlations",
            height=300,
            xaxis_title="Factors",
            yaxis_title="Factors"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_factor_matrices(data_loader, sidebar_state):
    """Render factor matrices when available"""
    
    st.subheader("Factor Risk Matrix")
    
    # NEW: Check for matrix data using hierarchical schema
    schema_data = data_loader.get_comprehensive_schema_data(sidebar_state.selected_node)
    factor_risk_matrix = None
    
    if schema_data:
        matrices_section = schema_data.get('matrices', {})
        factor_risk_matrix = matrices_section.get('factor_risk_contributions', {})
    
    if factor_risk_matrix:
        st.info("Factor risk contribution matrix visualization coming soon")
        
        # Show matrix dimensions
        if isinstance(factor_risk_matrix, dict):
            st.markdown(f"**Matrix available with {len(factor_risk_matrix)} entries**")
        
    else:
        st.info("Factor risk matrix not available - will populate when computed")
        st.markdown("""
        **Expected Matrix Structure:**
        
        The factor risk matrix will show risk contributions broken down by:
        - **Rows**: Portfolio components/assets
        - **Columns**: Risk factors
        - **Values**: Risk contribution from each factor to each component
        
        This provides detailed insight into how different factors drive risk
        at the component level across the portfolio hierarchy.
        """)
    
    # Show what matrix data might be available using hierarchical schema
    if schema_data:
        matrices_section = schema_data.get('matrices', {})
        arrays_section = schema_data.get('arrays', {})
        
        available_matrices = []
        if matrices_section:
            available_matrices.extend(matrices_section.keys())
        if arrays_section:
            available_matrices.extend([f"{k} (array)" for k in arrays_section.keys()])
        
        if available_matrices:
            st.markdown("**Available Matrix/Array Data:**")
            for matrix in available_matrices:
                st.markdown(f"• {matrix} (visualization coming soon)")
        else:
            st.markdown("**No matrix data currently available**")
    else:
        st.markdown("**No schema data available for matrix analysis**")