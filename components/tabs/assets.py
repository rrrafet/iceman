import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color

def render_assets_tab(data_loader, sidebar_state):
    """Render Tab 6 - Assets (detail)"""
    
    st.header("Assets - Detailed Component Analysis")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # Master table
    render_master_table(data_loader, sidebar_state)
    
    # Right drawer is handled through streamlit's native selection mechanism

def render_master_table(data_loader, sidebar_state):
    """Render master table of components with all key metrics"""
    
    st.subheader("Component Master Table")
    
    current_lens = sidebar_state.lens
    current_node = sidebar_state.selected_node
    
    # Get available components
    available_components = data_loader.get_available_hierarchical_components()
    
    if not available_components:
        st.info("No components available")
        return
    
    # Get hierarchy information
    hierarchy_info = data_loader.get_hierarchy_info()
    component_metadata = hierarchy_info.get('component_metadata', {})
    
    # Get weights data
    portfolio_weights = data_loader.get_weights("portfolio_weights")
    benchmark_weights = data_loader.get_weights("benchmark_weights")
    active_weights = data_loader.get_weights("active_weights")
    
    # Get contributions data
    contributions = data_loader.get_contributions(current_lens, "by_asset")
    
    # Build table data
    table_data = []
    
    for component_id in available_components:
        metadata = component_metadata.get(component_id, {})
        
        # Get weights
        port_weight = portfolio_weights.get(component_id, 0.0)
        bench_weight = benchmark_weights.get(component_id, 0.0)
        active_weight = active_weights.get(component_id, 0.0)
        
        # Get contribution
        contribution = contributions.get(component_id, 0.0)
        
        # Calculate flags
        flags = []
        if active_weight < 0:
            flags.append("Negative Weight")
        if abs(contribution) > 50:  # Threshold for high risk contribution
            flags.append("High Risk")
        if abs(active_weight) > 2.0:  # Threshold for large position
            flags.append("Large Position")
        
        table_data.append({
            'Component ID': component_id,
            'Type': metadata.get('type', 'N/A'),
            'Level': metadata.get('level', 0),
            'Portfolio Weight (%)': port_weight,
            'Benchmark Weight (%)': bench_weight,
            'Active Weight (%)': active_weight,
            'Risk Contribution (bps)': contribution,
            'Flags': ', '.join(flags) if flags else 'None'
        })
    
    if not table_data:
        st.info("No component data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(table_data)
    
    # Format numerical columns
    df['Portfolio Weight (%)'] = df['Portfolio Weight (%)'].round(2)
    df['Benchmark Weight (%)'] = df['Benchmark Weight (%)'].round(2)
    df['Active Weight (%)'] = df['Active Weight (%)'].round(2)
    df['Risk Contribution (bps)'] = df['Risk Contribution (bps)'].round(1)
    
    # Add sorting and filtering options
    render_table_controls(df)
    
    st.divider()
    
    # Display the filtered and sorted table
    filtered_df = apply_table_filters(df)
    
    if filtered_df.empty:
        st.info("No components match the current filters")
        return
    
    # Add row selection
    selected_rows = st.data_editor(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Component ID': st.column_config.TextColumn(
                "Component ID",
                help="Click to view details",
                width="medium"
            ),
            'Portfolio Weight (%)': st.column_config.NumberColumn(
                "Portfolio Weight (%)",
                format="%.2f",
                width="small"
            ),
            'Benchmark Weight (%)': st.column_config.NumberColumn(
                "Benchmark Weight (%)", 
                format="%.2f",
                width="small"
            ),
            'Active Weight (%)': st.column_config.NumberColumn(
                "Active Weight (%)",
                format="%.2f", 
                width="small"
            ),
            'Risk Contribution (bps)': st.column_config.NumberColumn(
                "Risk Contribution (bps)",
                format="%.1f",
                width="small"
            )
        },
        key="assets_table"
    )
    
    # Component drawer
    render_component_drawer(data_loader, sidebar_state, filtered_df)

def render_table_controls(df):
    """Render controls for sorting and filtering the table"""
    
    st.markdown("**Table Controls**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sort options
        sort_column = st.selectbox(
            "Sort by",
            options=['Component ID', 'Portfolio Weight (%)', 'Benchmark Weight (%)', 
                    'Active Weight (%)', 'Risk Contribution (bps)', 'Level'],
            index=0,
            key="assets_sort_column"
        )
        
        sort_ascending = st.checkbox(
            "Ascending",
            value=True,
            key="assets_sort_ascending"
        )
    
    with col2:
        # Filter by type
        available_types = df['Type'].unique().tolist()
        selected_types = st.multiselect(
            "Filter by Type",
            options=available_types,
            default=available_types,
            key="assets_type_filter"
        )
    
    with col3:
        # Filter by level
        available_levels = sorted(df['Level'].unique().tolist())
        selected_levels = st.multiselect(
            "Filter by Level", 
            options=available_levels,
            default=available_levels,
            key="assets_level_filter"
        )
    
    # Additional filters
    col4, col5 = st.columns(2)
    
    with col4:
        # Weight threshold filter
        min_abs_weight = st.number_input(
            "Min Absolute Active Weight (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            key="assets_weight_filter"
        )
    
    with col5:
        # Risk threshold filter
        min_abs_risk = st.number_input(
            "Min Absolute Risk Contribution (bps)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key="assets_risk_filter"
        )

def apply_table_filters(df):
    """Apply filters to the DataFrame"""
    
    filtered_df = df.copy()
    
    # Type filter
    selected_types = st.session_state.get("assets_type_filter", [])
    if selected_types:
        filtered_df = filtered_df[filtered_df['Type'].isin(selected_types)]
    
    # Level filter
    selected_levels = st.session_state.get("assets_level_filter", [])
    if selected_levels:
        filtered_df = filtered_df[filtered_df['Level'].isin(selected_levels)]
    
    # Weight filter
    min_abs_weight = st.session_state.get("assets_weight_filter", 0.0)
    if min_abs_weight > 0:
        filtered_df = filtered_df[abs(filtered_df['Active Weight (%)']) >= min_abs_weight]
    
    # Risk filter
    min_abs_risk = st.session_state.get("assets_risk_filter", 0.0)
    if min_abs_risk > 0:
        filtered_df = filtered_df[abs(filtered_df['Risk Contribution (bps)']) >= min_abs_risk]
    
    # Sort
    sort_column = st.session_state.get("assets_sort_column", "Component ID")
    sort_ascending = st.session_state.get("assets_sort_ascending", True)
    
    if sort_column in filtered_df.columns:
        filtered_df = filtered_df.sort_values(
            sort_column, 
            ascending=sort_ascending
        )
    
    return filtered_df.reset_index(drop=True)

def render_component_drawer(data_loader, sidebar_state, df):
    """Render component details drawer"""
    
    st.divider()
    st.subheader("Component Details")
    
    if df.empty:
        st.info("No components to display")
        return
    
    # Component selector
    selected_component = st.selectbox(
        "Select component for detailed view",
        options=df['Component ID'].tolist(),
        index=0,
        key="component_detail_selector"
    )
    
    if not selected_component:
        return
    
    # Get component details
    component_row = df[df['Component ID'] == selected_component].iloc[0]
    
    # Display component overview
    render_component_overview(component_row, data_loader, sidebar_state, selected_component)
    
    st.divider()
    
    # Component specific charts
    col1, col2 = st.columns(2)
    
    with col1:
        render_component_risk_breakdown(data_loader, sidebar_state, selected_component)
    
    with col2:
        render_component_time_series(data_loader, sidebar_state, selected_component)

def render_component_overview(component_row, data_loader, sidebar_state, component_id):
    """Render overview metrics for selected component"""
    
    st.markdown(f"**Component Overview: {component_id}**")
    
    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Type/Level",
            f"{component_row['Type']}/L{component_row['Level']}"
        )
    
    with col2:
        port_weight = component_row['Portfolio Weight (%)']
        st.metric("Portfolio Weight", f"{port_weight:.2f}%")
    
    with col3:
        active_weight = component_row['Active Weight (%)']
        color = "inverse" if active_weight < 0 else "normal"
        st.metric(
            "Active Weight", 
            f"{active_weight:.2f}%",
            delta=f"vs benchmark"
        )
    
    with col4:
        risk_contrib = component_row['Risk Contribution (bps)']
        st.metric("Risk Contribution", f"{risk_contrib:.1f} bps")
    
    # Flags and alerts
    flags = component_row.get('Flags', 'None')
    if flags != 'None':
        st.warning(f"Flags: {flags}")
    
    # Get core metrics for this component if available
    core_metrics = data_loader.get_core_metrics(sidebar_state.lens, component_id)
    
    if core_metrics:
        st.markdown("**Risk Metrics:**")
        col5, col6, col7 = st.columns(3)
        
        with col5:
            total_risk = core_metrics.get('total_risk', 0)
            st.metric("Total Risk", f"{total_risk:.0f} bps")
        
        with col6:
            factor_contrib = core_metrics.get('factor_risk_contribution', 0)
            st.metric("Factor Risk", f"{factor_contrib:.0f} bps")
        
        with col7:
            specific_contrib = core_metrics.get('specific_risk_contribution', 0)
            st.metric("Specific Risk", f"{specific_contrib:.0f} bps")

def render_component_risk_breakdown(data_loader, sidebar_state, component_id):
    """Render risk breakdown chart for component"""
    
    st.markdown("**Risk Breakdown**")
    
    # NEW: Get factor contributions for this component using hierarchical schema
    factor_contribs = data_loader.get_factor_contributions_from_schema(component_id, sidebar_state.lens)
    
    if factor_contribs:
        # Show top 5 factor contributors
        # Convert to basis points for display
        factor_contribs_bps = {f: v * 10000 for f, v in factor_contribs.items()}
        
        sorted_contribs = sorted(
            factor_contribs_bps.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        if sorted_contribs:
            factors, contributions = zip(*sorted_contribs)
            
            fig = go.Figure(go.Bar(
                x=list(contributions),
                y=list(factors),
                orientation='h',
                marker_color=[
                    get_chart_color("positive") if v >= 0 else get_chart_color("negative")
                    for v in contributions
                ],
                text=[f"{v:.1f} bps" for v in contributions],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"Top Factor Contributors - {component_id}",
                xaxis_title="Contribution (bps)",
                yaxis_title="Factors",
                height=250,
                showlegend=False
            )
            
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No factor contribution data")
    else:
        st.info("No factor breakdown available for this component")

def render_component_time_series(data_loader, sidebar_state, component_id):
    """Render small sparkline time series for component"""
    
    st.markdown("**Returns Time Series**")
    
    # Get time series data for this component
    component_data = data_loader.get_time_series_data('portfolio_returns', component_id)
    
    if component_data:
        # Apply date range filter
        start_idx, end_idx = sidebar_state.date_range
        if start_idx > 0 or end_idx < len(component_data) - 1:
            component_data = component_data[start_idx:end_idx + 1]
        
        periods = list(range(len(component_data)))
        
        # Create sparkline
        fig = go.Figure(go.Scatter(
            x=periods,
            y=component_data,
            mode='lines',
            name='Returns',
            line=dict(color=get_chart_color("portfolio"), width=2),
            hovertemplate='Period: %{x}<br>Return: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Returns Sparkline",
            xaxis_title="Period",
            yaxis_title="Return",
            height=250,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show basic stats
        mean_return = sum(component_data) / len(component_data)
        volatility = (sum([(x - mean_return)**2 for x in component_data]) / len(component_data))**0.5
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Return", f"{mean_return:.4f}")
        with col2:
            st.metric("Volatility", f"{volatility:.4f}")
    
    else:
        st.info(f"No time series data available for {component_id}")
        st.markdown("Time series will show when available from time_series.*_returns[component]")