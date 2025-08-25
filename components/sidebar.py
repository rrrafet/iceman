import streamlit as st
from data_loader import SidebarState
from components.data_refresh import render_data_refresh_controls

def render_sidebar(data_loader) -> SidebarState:
    """Render global sidebar with filters and return state"""
    
    with st.sidebar:
        st.header("🎯 Maverick Controls")
        
        # Snapshot metadata display
        st.subheader("Snapshot")
        metadata = data_loader.data.get('metadata', {})
        st.text(f"Analysis: {metadata.get('analysis_type', 'N/A')}")
        st.text(f"Frequency: {metadata.get('data_frequency', 'N/A')}")
        st.text(f"Schema: v{metadata.get('schema_version', 'N/A')}")
        
        st.divider()
        
        # Lens selector
        st.subheader("📊 Lens")
        lens = st.selectbox(
            "Select view perspective",
            options=["portfolio", "benchmark", "active"],
            index=0,
            format_func=lambda x: x.title(),
            key="lens_selector"
        )
        
        # Node selector (tree-aware)
        st.subheader("🏗️ Node Selector")
        component_names = data_loader.get_component_names()
        
        # Default to TOTAL if available
        default_node = "TOTAL" if "TOTAL" in component_names else (component_names[0] if component_names else "TOTAL")
        
        selected_node = st.selectbox(
            "Navigate hierarchy",
            options=component_names if component_names else ["TOTAL"],
            index=component_names.index(default_node) if default_node in component_names else 0,
            key="node_selector"
        )
        
        # Show component metadata if available
        hierarchy = data_loader.get_hierarchy_info()
        component_metadata = hierarchy.get('component_metadata', {})
        if selected_node in component_metadata:
            metadata = component_metadata[selected_node]
            st.caption(f"Type: {metadata.get('type', 'N/A')} | Level: {metadata.get('level', 'N/A')}")
        
        st.divider()
        
        # Date range selector
        st.subheader("📅 Date Range")
        time_series_meta = data_loader.data.get('time_series', {}).get('metadata', {})
        start_date = time_series_meta.get('start_date', '2023-01-01')
        end_date = time_series_meta.get('end_date', '2024-12-31')
        
        st.text(f"Available: {start_date} to {end_date}")
        
        # Get data length for slider
        sample_data = data_loader.get_time_series_data('portfolio_returns', 'TOTAL')
        max_periods = len(sample_data) if sample_data else 60
        
        date_range = st.slider(
            "Select period range",
            min_value=0,
            max_value=max_periods - 1,
            value=(0, max_periods - 1),
            key="date_range_slider"
        )
        
        # Preset buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("1Y", key="1y_preset"):
                st.session_state.date_range_slider = (max(0, max_periods - 252), max_periods - 1)
                st.rerun()
        with col2:
            if st.button("3Y", key="3y_preset"):
                st.session_state.date_range_slider = (max(0, max_periods - 756), max_periods - 1)
                st.rerun()
        with col3:
            if st.button("All", key="all_preset"):
                st.session_state.date_range_slider = (0, max_periods - 1)
                st.rerun()
        
        st.divider()
        
        # Factor filter
        st.subheader("🎛️ Factor Filter")
        factor_names = data_loader.get_factor_names()
        
        selected_factors = st.multiselect(
            "Select factors to analyze",
            options=factor_names,
            default=[],
            key="factor_filter"
        )
        
        if st.button("Select All Factors", key="select_all_factors"):
            st.session_state.factor_filter = factor_names.copy()
            st.rerun()
        
        if st.button("Clear All", key="clear_all_factors"):
            st.session_state.factor_filter = []
            st.rerun()
        
        st.divider()
        
        # Display options
        st.subheader("🎨 Display")
        annualized = st.toggle("Annualized", value=True, key="annualized_toggle")
        show_percentage = st.toggle("Show % of total", value=True, key="percentage_toggle") 
        
        currency = data_loader.data.get('time_series', {}).get('currency', 'USD')
        st.text(f"Currency: {currency}")
        
        st.divider()
        
        # Data refresh controls
        st.divider()
        render_data_refresh_controls(data_loader)
        
        st.divider()
        
        # Downloads section
        st.subheader("💾 Downloads")
        if st.button("Current View JSON", key="download_current"):
            st.info("Download functionality coming soon")
        if st.button("Full Payload", key="download_full"):
            st.info("Download functionality coming soon")
        
        # Validation status
        validation = data_loader.get_validation_info()
        checks = validation.get('checks', {})
        passes = checks.get('passes', False)
        
        st.divider()
        st.subheader("✅ Status")
        if passes:
            st.success("All checks passed")
        else:
            st.warning("Some checks failed")
    
    return SidebarState(
        lens=lens,
        selected_node=selected_node,
        date_range=date_range,
        selected_factors=selected_factors,
        annualized=annualized,
        show_percentage=show_percentage
    )