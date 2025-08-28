import streamlit as st
from data_loader import SidebarState

def render_sidebar(data_loader) -> SidebarState:
    """Render simplified sidebar using direct schema data access"""
    
    with st.sidebar:
        st.header("Maverick Controls")
        
        # Current configuration display
        st.subheader("Configuration")
        if data_loader.current_config_id:
            st.text(f"Config: {data_loader.current_config_id}")
        else:
            st.text("Config: Not loaded")
        
        st.divider()
        
        # Lens selector
        st.subheader("Lens")
        lens = st.selectbox(
            "Select view perspective",
            options=["portfolio", "benchmark", "active"],
            index=0,
            format_func=lambda x: x.title(),
            key="lens_selector"
        )
        
        # Node selector
        st.subheader("Node Navigator")
        
        # Get all available components from schema
        available_components = data_loader.get_all_components()
        
        # Default to TOTAL if available
        default_node = "TOTAL" if "TOTAL" in available_components else (available_components[0] if available_components else "TOTAL")
        
        selected_node = st.selectbox(
            "Navigate hierarchy",
            options=available_components if available_components else ["TOTAL"],
            index=available_components.index(default_node) if default_node in available_components else 0,
            key="node_selector"
        )
        
        st.divider()
        
        # Date range selector (simplified)
        st.subheader("Date Range")
        
        date_range = st.slider(
            "Select period range",
            min_value=0,
            max_value=100,
            value=(0, 100),
            key="date_range_slider"
        )
        
        st.divider()
        
        # Factor filter
        st.subheader("Factor Filter")
        factor_names = data_loader.get_factor_names()
        
        selected_factors = st.multiselect(
            "Select factors to analyze",
            options=factor_names,
            default=[],
            key="factor_filter"
        )
        
        st.divider()
        
        # Display options
        st.subheader("Display")
        annualized = st.toggle("Annualized", value=True, key="annualized_toggle")
        show_percentage = st.toggle("Show % of total", value=True, key="percentage_toggle")
    
    return SidebarState(
        lens=lens,
        selected_node=selected_node,
        date_range=date_range,
        selected_factors=selected_factors,
        annualized=annualized,
        show_percentage=show_percentage
    )