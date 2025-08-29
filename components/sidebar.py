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
        
        # Default to root component if available
        root_component = data_loader.root_component_id
        default_node = root_component if root_component in available_components else (available_components[0] if available_components else root_component)
        
        selected_node = st.selectbox(
            "Navigate hierarchy",
            options=available_components if available_components else [root_component],
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
        
        # Risk Model selector
        st.subheader("Risk Model")
        available_models = data_loader.get_available_risk_models()
        
        if available_models:
            model_options = [model['id'] for model in available_models]
            model_labels = {model['id']: model['name'] for model in available_models}
            
            # Default to current risk model or first available
            default_model = data_loader.current_risk_model or model_options[0]
            default_index = model_options.index(default_model) if default_model in model_options else 0
            
            selected_risk_model = st.selectbox(
                "Select risk model",
                options=model_options,
                index=default_index,
                format_func=lambda x: model_labels[x],
                key="risk_model_selector"
            )
            
            # Check if risk model changed and switch if needed
            if hasattr(st.session_state, 'previous_risk_model'):
                if st.session_state.previous_risk_model != selected_risk_model:
                    with st.spinner("Switching risk model..."):
                        success = data_loader.switch_risk_model(selected_risk_model)
                        if success:
                            st.success(f"Switched to {model_labels[selected_risk_model]}")
                        else:
                            st.error(f"Failed to switch to {model_labels[selected_risk_model]}")
            
            st.session_state.previous_risk_model = selected_risk_model
        else:
            selected_risk_model = "default"
        
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
        selected_risk_model=selected_risk_model,
        annualized=annualized,
        show_percentage=show_percentage
    )