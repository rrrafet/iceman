import streamlit as st
from dataclasses import dataclass
from typing import List

@dataclass
class SidebarState:
    """State object containing sidebar filter selections."""
    lens: str
    selected_component_id: str
    selected_risk_model: str
    selected_factors: List[str]
    annualized: bool
    show_percentage: bool

def render_sidebar(config_service, data_access_service) -> SidebarState:
    """Render sidebar using 3-layer architecture services."""
    
    with st.sidebar:
        st.header("Maverick Controls")
        
        # Current configuration display
        st.subheader("Configuration")
        portfolio_name = config_service.get_portfolio_name()
        st.text(f"Graph: {portfolio_name}")
        
        st.divider()
        
        # Risk Model selector
        st.subheader("Risk Model")
        available_models = config_service.get_available_risk_models()
        default_risk_model = config_service.get_default_risk_model()
        
        # Find default index
        try:
            default_index = available_models.index(default_risk_model)
        except (ValueError, AttributeError):
            default_index = 0
        
        selected_risk_model = st.selectbox(
            "Factor risk model",
            options=available_models,
            index=default_index,
            key="risk_model_selector"
        )
        
        st.divider()
        
        # Component selector
        st.subheader("Component")
        
        # Get all available components
        available_components = data_access_service.get_all_component_ids()
        if not available_components:
            available_components = [config_service.get_root_component_id()]
        
        # Default to root component
        root_component = config_service.get_root_component_id()
        default_component = root_component if root_component in available_components else (available_components[0] if available_components else root_component)
        
        try:
            default_component_index = available_components.index(default_component)
        except ValueError:
            default_component_index = 0
        
        selected_component_id = st.selectbox(
            "Select component",
            options=available_components,
            index=default_component_index,
            key="component_selector"
        )
        
        st.divider()
        
        # Lens selector
        st.subheader("Lens")
        lens_options = ["portfolio", "benchmark", "active"]
        default_lens = config_service.get_default_lens()
        
        try:
            default_lens_index = lens_options.index(default_lens)
        except ValueError:
            default_lens_index = 0
        
        lens = st.selectbox(
            "Select view perspective",
            options=lens_options,
            index=default_lens_index,
            format_func=lambda x: x.title(),
            key="lens_selector"
        )
        
        # st.divider()
        
        # Factor filter  
        # st.subheader("Factor Filter")
        # try:
        #     factor_names = data_access_service.get_available_factors()
        # except:
        #     factor_names = []
        
        # selected_factors = st.multiselect(
        #     "Select factors to analyze",
        #     options=factor_names,
        #     default=[],
        #     key="factor_filter"
        # )
        
        # st.divider()
        
        # Display options
        # st.subheader("Display")
        # nnualized = st.toggle("Annualized", value=config_service.get_annualized_default(), key="annualized_toggle")
        # show_percentage = st.toggle("Show % of total", value=True, key="percentage_toggle")
    
    return SidebarState(
        lens=lens,
        selected_component_id=selected_component_id,
        selected_risk_model=selected_risk_model,
        selected_factors=None,
        annualized=None,
        show_percentage=None
    )