import streamlit as st
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color, get_factor_color

def render_hierarchy_explorer_tab(data_loader, sidebar_state):
    """Render Tab 3 - Hierarchy Explorer (drilldown)"""
    
    st.header("Hierarchy Explorer - Component Drilldown")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # Two columns: tree on left, node context on right
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Left pane: expandable tree
        render_hierarchy_tree(data_loader, sidebar_state)
    
    with col2:
        # Right pane: node context and details
        render_node_context(data_loader, sidebar_state)

def render_hierarchy_tree(data_loader, sidebar_state):
    """Render expandable tree structure"""
    
    st.subheader("Portfolio Tree")
    
    # Get hierarchy information
    hierarchy_info = data_loader.get_hierarchy_info()
    component_metadata = hierarchy_info.get('component_metadata', {})
    
    if not component_metadata:
        st.info("No hierarchy information available")
        return
    
    # Build tree structure organized by levels
    levels = {}
    for comp_id, metadata in component_metadata.items():
        level = metadata.get('level', 0)
        if level not in levels:
            levels[level] = []
        levels[level].append({
            'id': comp_id,
            'name': metadata.get('name', comp_id),
            'type': metadata.get('type', 'node')
        })
    
    # Display tree by levels
    current_node = sidebar_state.selected_node
    
    for level in sorted(levels.keys()):
        components = levels[level]
        
        if level == 0:
            # Root level - always show
            for component in components:
                is_selected = component['id'] == current_node
                
                if is_selected:
                    st.markdown(f"**→ {component['name']}** ({component['type']})")
                else:
                    if st.button(f"{component['name']}", key=f"tree_{component['id']}", help=f"Navigate to {component['id']}"):
                        # Update selected node (would need session state handling)
                        st.info(f"Selected: {component['id']}")
        else:
            # Child levels - show if parent is selected or expanded
            # For now, show all components for simplicity
            st.markdown(f"**Level {level}:**")
            for component in components:
                is_selected = component['id'] == current_node
                indent = "  " * level
                
                if is_selected:
                    st.markdown(f"{indent}**→ {component['name']}** ({component['type']})")
                else:
                    if st.button(f"{indent}{component['name']}", key=f"tree_{component['id']}", help=f"Navigate to {component['id']}"):
                        st.info(f"Selected: {component['id']}")

def render_node_context(data_loader, sidebar_state):
    """Render right pane with node context and details"""
    
    current_node = sidebar_state.selected_node
    current_lens = sidebar_state.lens
    
    st.subheader(f"Node Context: {current_node}")
    
    # Mini KPI cards for this node
    render_mini_kpi_cards(data_loader, sidebar_state, current_node, current_lens)
    
    st.divider()
    
    # Children table
    render_children_table(data_loader, sidebar_state, current_node)
    
    st.divider()
    
    # Mini treemap of child contributions
    render_mini_treemap(data_loader, sidebar_state, current_node)
    
    st.divider()
    
    # Exposures at this node
    render_node_exposures(data_loader, sidebar_state, current_node, current_lens)
    
    st.divider()
    
    # Navigation buttons
    render_navigation_buttons(data_loader, current_node)

def render_mini_kpi_cards(data_loader, sidebar_state, node_id, lens):
    """Render mini KPI cards for the selected node"""
    
    st.markdown("**Risk Metrics**")
    
    # Get core metrics for this node
    metrics = data_loader.get_core_metrics(lens, node_id)
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_risk = metrics.get('total_risk', 0)
            st.metric("Total Risk", f"{total_risk:.0f} bps")
        
        with col2:
            factor_contrib = metrics.get('factor_risk_contribution', 0)
            st.metric("Factor Risk", f"{factor_contrib:.0f} bps")
        
        with col3:
            specific_contrib = metrics.get('specific_risk_contribution', 0)
            st.metric("Specific Risk", f"{specific_contrib:.0f} bps")
        
        with col4:
            factor_pct = metrics.get('factor_risk_percentage', 0)
            st.metric("Factor %", f"{factor_pct:.1f}%")
    else:
        st.info(f"No risk metrics available for {node_id}")

def render_children_table(data_loader, sidebar_state, node_id):
    """Render table of child components"""
    
    st.markdown("**Child Components**")
    
    # Get children of current node
    children = data_loader.get_drilldown_options(node_id)
    
    if not children:
        st.info(f"No child components found for {node_id}")
        return
    
    # Get hierarchy metadata
    hierarchy_info = data_loader.get_hierarchy_info()
    component_metadata = hierarchy_info.get('component_metadata', {})
    
    # Get weights and contributions for children
    weights = data_loader.get_weights("portfolio_weights")
    contributions = data_loader.get_component_asset_contributions(node_id, sidebar_state.lens)
    
    # Build table data
    table_data = []
    for child_id in children:
        metadata = component_metadata.get(child_id, {})
        weight = weights.get(child_id, 0.0)
        contribution = contributions.get(child_id, 0.0)
        
        table_data.append({
            'Component': child_id,
            'Type': metadata.get('type', 'N/A'),
            'Level': metadata.get('level', 'N/A'),
            'Weight (%)': f"{weight:.2f}",
            'Risk Contrib (bps)': f"{contribution:.0f}"
        })
    
    if table_data:
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No child component data available")

def render_mini_treemap(data_loader, sidebar_state, node_id):
    """Render mini treemap of child contributions"""
    
    st.markdown("**Child Risk Contributions**")
    
    # Get children and their contributions
    children = data_loader.get_drilldown_options(node_id)
    contributions = data_loader.get_component_asset_contributions(node_id, sidebar_state.lens)
    
    if not children or not contributions:
        st.info("No contribution data for visualization")
        return
    
    # Filter contributions for children only
    child_contribs = {child: contributions.get(child, 0) for child in children if child in contributions}
    
    if not child_contribs:
        st.info("No child contributions available")
        return
    
    # Create simple bar chart instead of treemap for now
    names = list(child_contribs.keys())
    values = list(child_contribs.values())
    
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color=[get_chart_color("positive") if v >= 0 else get_chart_color("negative") for v in values],
        text=[f"{v:.0f} bps" for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Child Risk Contributions",
        xaxis_title="Contribution (bps)",
        yaxis_title="Components",
        height=max(200, len(child_contribs) * 25 + 100),
        showlegend=False
    )
    
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

def render_node_exposures(data_loader, sidebar_state, node_id, lens):
    """Render factor exposures at this node"""
    
    st.markdown("**Factor Exposures**")
    
    # Get exposures for this node/lens
    exposures = data_loader.get_exposures(lens)
    
    if not exposures:
        st.info(f"No factor exposures available for {lens} lens")
        return
    
    # Filter by selected factors if any
    if sidebar_state.selected_factors:
        exposures = data_loader.filter_data_by_factors(exposures, sidebar_state.selected_factors)
    
    if not exposures:
        st.info("No exposures to display")
        return
    
    # Create horizontal bar chart
    factors = list(exposures.keys())
    values = list(exposures.values())
    
    fig = go.Figure(go.Bar(
        x=values,
        y=factors,
        orientation='h',
        marker_color=[get_factor_color(f) for f in factors],
        text=[f"{v:.3f}" for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Factor Exposures - {lens.title()}",
        xaxis_title="Exposure",
        yaxis_title="Factors",
        height=max(200, len(factors) * 20 + 100),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_navigation_buttons(data_loader, node_id):
    """Render up/down navigation buttons"""
    
    st.markdown("**Navigation**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Up button - navigate to parent
        if data_loader.can_drill_up(node_id):
            parent = data_loader.get_component_parent(node_id)
            if st.button(f"↑ Up to {parent}", key="nav_up"):
                st.info(f"Navigate to parent: {parent}")
        else:
            st.button("↑ Up (unavailable)", disabled=True, key="nav_up_disabled")
    
    with col2:
        # Refresh button
        if st.button("Refresh", key="nav_refresh"):
            st.info("Refreshing node data...")
    
    with col3:
        # Children count info
        children = data_loader.get_drilldown_options(node_id)
        st.metric("Children", len(children))