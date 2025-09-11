import streamlit as st
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd

@dataclass
class SidebarState:
    """State object containing sidebar filter selections."""
    lens: str
    selected_component_id: str
    selected_risk_model: str
    selected_portfolio_graph: str
    selected_factors: List[str]
    annualized: bool
    show_percentage: bool
    frequency: str
    date_range_start: Optional[datetime]
    date_range_end: Optional[datetime]
    date_range_preset: str

def render_sidebar(config_service, data_access_service) -> SidebarState:
    """Render sidebar using 3-layer architecture services."""
    
    with st.sidebar:
        st.header("Maverick Controls")
        
        # Current configuration display
        st.subheader("Configuration")
        portfolio_name = config_service.get_portfolio_name()
        st.text(f"Graph: {portfolio_name}")
        
        st.divider()
        
        # Portfolio Graph selector
        st.subheader("Portfolio Graph")
        available_portfolio_graphs = config_service.get_available_portfolio_graphs()
        
        # Get current portfolio graph from configuration
        current_portfolio_graph = config_service.get_default_portfolio_graph()
        
        # Find current index
        try:
            default_portfolio_index = available_portfolio_graphs.index(current_portfolio_graph)
        except (ValueError, AttributeError):
            default_portfolio_index = 0
        
        selected_portfolio_graph = st.selectbox(
            "Select portfolio graph",
            options=available_portfolio_graphs,
            index=default_portfolio_index,
            key="portfolio_graph_selector",
            help="Select a portfolio configuration to analyze"
        )
        
        st.divider()
        
        # Risk Model selector
        st.subheader("Risk Model")
        available_models = config_service.get_available_risk_models()
        
        # Get current risk model from data access service
        current_risk_model = data_access_service.get_current_risk_model()
        if not current_risk_model:
            current_risk_model = config_service.get_default_risk_model()
        
        # Find current index
        try:
            default_index = available_models.index(current_risk_model)
        except (ValueError, AttributeError):
            default_index = 0
        
        selected_risk_model = st.selectbox(
            "Factor risk model",
            options=available_models,
            index=default_index,
            key="risk_model_selector",
            help="Select a factor risk model to analyze portfolio risk"
        )
        
        st.divider()
        
        # Component selector
        st.subheader("Component")
        
        # Get all available components
        available_components = data_access_service.get_all_component_ids()
        portfolio_graph = data_access_service.risk_analysis_service.get_portfolio_graph()
        if not available_components:
            available_components = [config_service.get_root_component_id(portfolio_graph)]
        
        # Default to root component
        root_component = config_service.get_root_component_id(portfolio_graph)
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
        
        st.divider()
        
        # Frequency selector
        st.subheader("Data Frequency")
        frequency_options = ["D", "B", "W-FRI", "ME"]
        frequency_labels = {
            "D": "Daily", 
            "B": "Business Daily (Native)",
            "W-FRI": "Weekly (Friday)", 
            "ME": "Monthly"
        }
        
        # Default to weekly
        default_frequency = "W-FRI"
        try:
            default_frequency_index = frequency_options.index(default_frequency)
        except ValueError:
            default_frequency_index = 2  # W-FRI is at index 2
        
        selected_frequency = st.selectbox(
            "Select data frequency",
            options=frequency_options,
            index=default_frequency_index,
            format_func=lambda x: frequency_labels.get(x, x),
            key="frequency_selector"
        )
        
        st.divider()
        
        # Date Range selector
        st.subheader("Date Range")
        
        def get_preset_dates(preset: str) -> tuple[datetime, datetime]:
            """Calculate start and end dates based on preset."""
            end_date = datetime.now()
            
            if preset == "Daily -6M":
                start_date = end_date - timedelta(days=180)
            elif preset == "Daily -1Y":
                start_date = end_date - timedelta(days=365)
            elif preset == "Daily -3Y":
                start_date = end_date - timedelta(days=365 * 3)
            elif preset == "Daily -5Y":
                start_date = end_date - timedelta(days=365 * 5)
            elif preset == "Weekly -1Y":
                start_date = end_date - timedelta(days=365)
            elif preset == "Weekly -3Y":
                start_date = end_date - timedelta(days=365 * 3)
            elif preset == "Weekly -5Y":
                start_date = end_date - timedelta(days=365 * 5)
            else:  # Custom
                start_date = end_date - timedelta(days=365)
            
            return start_date, end_date
        
        # Preset options
        preset_options = [
            "Daily -6M", "Daily -1Y", "Daily -3Y", "Daily -5Y",
            "Weekly -1Y", "Weekly -3Y", "Weekly -5Y", "Custom"
        ]
        
        # Get current data access service state to determine default
        current_start, current_end = data_access_service.get_date_range()
        
        # Initialize session state for date range if needed
        if "date_range_initialized" not in st.session_state:
            st.session_state.date_range_initialized = True
            # Set default preset
            if current_start is None and current_end is None:
                default_preset = "Weekly -3Y"
                default_start, default_end = get_preset_dates(default_preset)
                st.session_state.date_range_preset = default_preset
                st.session_state.date_range_start = default_start
                st.session_state.date_range_end = default_end
            else:
                # Use current values from data access service
                st.session_state.date_range_start = current_start
                st.session_state.date_range_end = current_end
                st.session_state.date_range_preset = "Custom"
        
        # Determine default preset index
        current_preset = getattr(st.session_state, 'date_range_preset', 'Weekly -3Y')
        try:
            default_preset_index = preset_options.index(current_preset)
        except ValueError:
            default_preset_index = 5  # Weekly -3Y is at index 5
        
        selected_preset = st.selectbox(
            "Select date range preset",
            options=preset_options,
            index=default_preset_index,
            key="date_range_preset_selector"
        )
        
        # Handle preset change
        if selected_preset != getattr(st.session_state, 'date_range_preset', 'Weekly -3Y'):
            st.session_state.date_range_preset = selected_preset
            if selected_preset != "Custom":
                # Update session state with preset dates
                preset_start, preset_end = get_preset_dates(selected_preset)
                st.session_state.date_range_start = preset_start
                st.session_state.date_range_end = preset_end
        
        # Calculate dates based on preset or use custom inputs
        if selected_preset == "Custom":
            # Get default values for custom inputs
            default_start = getattr(st.session_state, 'date_range_start', datetime.now() - timedelta(days=365*3))
            default_end = getattr(st.session_state, 'date_range_end', datetime.now())
            
            # Custom date inputs
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date", 
                    value=default_start.date() if isinstance(default_start, datetime) else default_start,
                    key="custom_start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=default_end.date() if isinstance(default_end, datetime) else default_end,
                    key="custom_end_date"
                )
            
            # Convert to datetime objects and update session state
            date_range_start = datetime.combine(start_date, datetime.min.time()) if start_date else None
            date_range_end = datetime.combine(end_date, datetime.min.time()) if end_date else None
            
            # Update session state
            st.session_state.date_range_start = date_range_start
            st.session_state.date_range_end = date_range_end
        else:
            # Use session state values for preset (already calculated above)
            date_range_start = st.session_state.date_range_start
            date_range_end = st.session_state.date_range_end
            
            # Display the calculated range
            if date_range_start and date_range_end:
                st.text(f"From: {date_range_start.strftime('%Y-%m-%d')}")
                st.text(f"To: {date_range_end.strftime('%Y-%m-%d')}")
        
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
        selected_portfolio_graph=selected_portfolio_graph,
        selected_factors=None,
        annualized=None,
        show_percentage=None,
        frequency=selected_frequency,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        date_range_preset=selected_preset
    )