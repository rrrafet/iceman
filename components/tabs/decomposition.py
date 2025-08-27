import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color

def render_decomposition_tab(data_loader, sidebar_state):
    """Render Tab 8 - Decomposition (allocation / selection)"""
    
    st.header("Decomposition - Allocation & Selection Effects")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # Check for decomposition data availability
    render_decomposition_status(data_loader)
    
    st.divider()
    
    # Placeholder design for when data is available
    render_decomposition_design(data_loader, sidebar_state)

def render_decomposition_status(data_loader):
    """Show status of decomposition data availability"""
    
    st.subheader("Decomposition Status")
    
    # Check for decomposition data in active lens
    try:
        active_data = data_loader.get_component_risk_analysis("TOTAL", "active")
        if active_data.get('success', False):
            data_dict = active_data.get('data', {})
            decomposition = data_dict.get('decomposition', {})
            
            allocation_effect = decomposition.get('allocation_effect', {})
            selection_effect = decomposition.get('selection_effect', {})
            interaction_effect = decomposition.get('interaction_effect', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if allocation_effect:
                    st.success("Allocation Effect: Available")
                    st.metric("Components", len(allocation_effect))
                else:
                    st.info("Allocation Effect: Not Available")
            
            with col2:
                if selection_effect:
                    st.success("Selection Effect: Available") 
                    st.metric("Components", len(selection_effect))
                else:
                    st.info("Selection Effect: Not Available")
            
            with col3:
                if interaction_effect:
                    st.success("Interaction Effect: Available")
                    st.metric("Components", len(interaction_effect))
                else:
                    st.info("Interaction Effect: Not Available")
        else:
            st.warning("Active risk analysis not available")
    except:
        st.warning("Unable to check decomposition status")
    
    # Explanation
    st.markdown("""
    **Decomposition Analysis Explanation:**
    
    Performance attribution decomposes active returns into three components:
    - **Allocation Effect**: Impact of over/underweighting sectors vs benchmark
    - **Selection Effect**: Impact of security selection within sectors
    - **Interaction Effect**: Combined impact of allocation and selection decisions
    
    This analysis helps identify whether excess returns come from sector allocation 
    decisions or individual security selection skills.
    """)

def render_decomposition_design(data_loader, sidebar_state):
    """Render placeholder design for decomposition analysis"""
    
    st.subheader("Decomposition Analysis")
    
    # Create tabs for different views
    decomp_tabs = st.tabs(["Summary", "By Factor", "By Component", "Waterfall"])
    
    with decomp_tabs[0]:
        render_decomposition_summary()
    
    with decomp_tabs[1]:
        render_decomposition_by_factor(data_loader, sidebar_state)
    
    with decomp_tabs[2]:
        render_decomposition_by_component(data_loader, sidebar_state)
    
    with decomp_tabs[3]:
        render_decomposition_waterfall(data_loader, sidebar_state)

def render_decomposition_summary():
    """Render summary view of decomposition effects"""
    
    st.markdown("**Effect Summary**")
    
    # Placeholder data structure
    st.info("""
    **When decomposition data is available, this section will show:**
    
    - Total active return broken down by effect
    - Statistical significance of each effect
    - Contribution to tracking error from each source
    - Time series evolution of effects
    """)
    
    # Mock structure for visualization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Allocation Effect**")
        st.metric("Total Impact", "--- bps", help="Coming soon")
        st.markdown("Top contributors will be listed here")
    
    with col2:
        st.markdown("**Selection Effect**") 
        st.metric("Total Impact", "--- bps", help="Coming soon")
        st.markdown("Top contributors will be listed here")
    
    with col3:
        st.markdown("**Interaction Effect**")
        st.metric("Total Impact", "--- bps", help="Coming soon")
        st.markdown("Combined effects will be shown here")

def render_decomposition_by_factor(data_loader, sidebar_state):
    """Render decomposition analysis by factor"""
    
    st.markdown("**Decomposition by Factor**")
    
    st.info("""
    **Factor-based Decomposition Analysis:**
    
    This view will break down allocation and selection effects by risk factors:
    
    - How much of the allocation effect comes from each factor exposure
    - Which factors contributed most to selection effects
    - Factor-specific interaction effects
    """)
    
    # Placeholder table structure
    st.markdown("**Expected Table Structure:**")
    
    import pandas as pd
    placeholder_data = pd.DataFrame({
        'Factor': ['Market', 'SMB', 'HML', 'Quality', 'Momentum'],
        'Allocation (bps)': ['---', '---', '---', '---', '---'],
        'Selection (bps)': ['---', '---', '---', '---', '---'], 
        'Interaction (bps)': ['---', '---', '---', '---', '---'],
        'Total (bps)': ['---', '---', '---', '---', '---']
    })
    
    st.dataframe(placeholder_data, use_container_width=True)

def render_decomposition_by_component(data_loader, sidebar_state):
    """Render decomposition analysis by component/sector"""
    
    st.markdown("**Decomposition by Component**")
    
    st.info("""
    **Component-based Decomposition Analysis:**
    
    This view will show allocation and selection effects for each portfolio component:
    
    - Sector/component level allocation decisions
    - Security selection within each sector
    - Component-specific interaction effects
    """)
    
    # Get available components for structure
    components = data_loader.get_available_hierarchical_components()[:5]  # Top 5 for example
    
    if components:
        st.markdown("**Expected Analysis Structure:**")
        
        import pandas as pd
        placeholder_data = pd.DataFrame({
            'Component': components,
            'Portfolio Weight (%)': ['---'] * len(components),
            'Benchmark Weight (%)': ['---'] * len(components),
            'Allocation (bps)': ['---'] * len(components),
            'Selection (bps)': ['---'] * len(components),
            'Interaction (bps)': ['---'] * len(components),
            'Total Effect (bps)': ['---'] * len(components)
        })
        
        st.dataframe(placeholder_data, use_container_width=True)
    else:
        st.markdown("Component structure will be shown when portfolio data is available")

def render_decomposition_waterfall(data_loader, sidebar_state):
    """Render waterfall chart for decomposition effects"""
    
    st.markdown("**Decomposition Waterfall**")
    
    st.info("""
    **Waterfall Chart Visualization:**
    
    The waterfall chart will show how each effect contributes to total active return:
    
    1. Starting point (benchmark return)
    2. + Allocation effect
    3. + Selection effect  
    4. + Interaction effect
    5. = Total portfolio return
    """)
    
    # Create placeholder waterfall structure
    fig = go.Figure()
    
    # Placeholder values
    categories = ['Benchmark', 'Allocation', 'Selection', 'Interaction', 'Portfolio']
    values = [0, 0, 0, 0, 0]  # Placeholder - would be actual decomposition values
    
    # This would be replaced with actual waterfall when data available
    fig.add_trace(go.Bar(
        x=categories,
        y=[5, 2, -1, 0.5, 6.5],  # Example values
        marker_color=['lightblue', 'green', 'red', 'orange', 'darkblue'],
        text=['Base', '+2 bps', '-1 bps', '+0.5 bps', 'Total'],
        textposition='outside',
        opacity=0.3  # Low opacity to show this is placeholder
    ))
    
    fig.update_layout(
        title="Example Decomposition Waterfall (Placeholder)",
        xaxis_title="Effect Type",
        yaxis_title="Return Contribution (bps)",
        height=400,
        showlegend=False
    )
    
    # Add note
    fig.add_annotation(
        text="Placeholder visualization - actual data coming soon",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="red"),
        bgcolor="white",
        bordercolor="red",
        borderwidth=2
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Interpretation Guide:**
    
    - **Positive Allocation**: Overweighting outperforming sectors
    - **Negative Allocation**: Overweighting underperforming sectors
    - **Positive Selection**: Picking outperforming securities within sectors
    - **Negative Selection**: Picking underperforming securities within sectors
    - **Interaction**: Amplification/dampening when allocation and selection align
    """)

def render_placeholder_message():
    """Render placeholder message for unavailable features"""
    
    st.warning("""
    **Decomposition Analysis - Coming Soon**
    
    This advanced attribution analysis requires:
    - Historical return data for portfolio and benchmark
    - Sector/component mappings
    - Performance attribution calculations
    
    The framework is ready - data integration is the next step.
    """)