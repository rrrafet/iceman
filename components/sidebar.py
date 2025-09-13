import streamlit as st
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)

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

        # Get current risk model
        current_risk_model = data_access_service.get_current_risk_model()

        if not current_risk_model:
            current_risk_model = config_service.get_default_risk_model()
        
        # Find current index
        try:
            default_index = available_models.index(current_risk_model) if current_risk_model in available_models else 0
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
        
        # Enhanced Frequency selector with ResamplingService integration
        st.subheader("Data Frequency")
        
        # Enhanced frequency options including quarterly and annual
        frequency_options = ["B", "W-FRI", "ME", "Q", "A"]
        frequency_labels = {
            "B": "Business Daily (Native) - No resampling",
            "W-FRI": "Weekly (Friday) - Compound returns", 
            "ME": "Monthly End - Compound returns",
            "Q": "Quarterly - Compound returns",
            "A": "Annual - Compound returns"
        }
        
        frequency_descriptions = {
            "B": "Native frequency - fastest processing, most granular data",
            "W-FRI": "Weekly resampling - good balance of detail and stability",
            "ME": "Monthly resampling - reduces noise, smoother trends", 
            "Q": "Quarterly resampling - long-term view, minimal noise",
            "A": "Annual resampling - highest level overview"
        }
        
        # Initialize session state for frequency if not exists
        if 'frequency_key' not in st.session_state:
            st.session_state.frequency_key = "W-FRI"  # Default to weekly
        
        # Use session state as single source of truth
        default_frequency = st.session_state.frequency_key
        
        # Ensure the stored frequency is valid
        if default_frequency not in frequency_options:
            default_frequency = "W-FRI"
            st.session_state.frequency_key = default_frequency
        
        try:
            default_frequency_index = frequency_options.index(default_frequency)
        except ValueError:
            default_frequency_index = 1  # W-FRI is at index 1
            st.session_state.frequency_key = "W-FRI"
        
        def on_frequency_change():
            """Callback to update session state when frequency changes."""
            st.session_state.frequency_key = st.session_state.frequency_selector
        
        selected_frequency = st.selectbox(
            "Select data frequency for analysis",
            options=frequency_options,
            index=default_frequency_index,
            format_func=lambda x: frequency_labels.get(x, x),
            key="frequency_selector",
            help="Choose how to aggregate return data. Higher frequencies provide more detail but may be noisier.",
            on_change=on_frequency_change
        )
        
        # Show frequency description
        st.caption(frequency_descriptions.get(selected_frequency, ""))
        
        # Show resampling status
        if selected_frequency == "B":
            st.success("Native frequency - no resampling required")
        
        st.divider()
        
        # Simplified Date Range selector  
        st.subheader("Analysis Period")
        
        def get_date_range(period_key: str) -> tuple[datetime, datetime]:
            """Calculate start and end dates for common analysis periods."""
            end_date = datetime.now()
            
            if period_key == "1Y":
                start_date = end_date - timedelta(days=365)
            elif period_key == "3Y":
                start_date = end_date - timedelta(days=365 * 3) 
            elif period_key == "5Y":
                start_date = end_date - timedelta(days=365 * 5)
            elif period_key == "ITD":
                # Use a very early date for inception-to-date
                start_date = datetime(2000, 1, 1)
            else:  # Custom
                start_date = end_date - timedelta(days=365 * 3)  # Default fallback
            
            return start_date, end_date
        
        # Simplified preset options focused on common analysis periods
        period_options = {
            "3Y": "Last 3 Years (Recommended)",
            "1Y": "Last 1 Year", 
            "5Y": "Last 5 Years",
            "ITD": "Inception to Date",
            "CUSTOM": "Custom Range"
        }
        
        period_descriptions = {
            "1Y": "Good for short-term analysis and recent performance trends",
            "3Y": "Balanced view capturing multiple market cycles - recommended for most analysis",
            "5Y": "Long-term view, smooths out market volatility, good for strategic analysis", 
            "ITD": "Full history available - may include very old, less relevant data",
            "CUSTOM": "Specify exact date range for targeted analysis"
        }
        
        # Initialize session state for period preset if not exists
        if 'period_preset_key' not in st.session_state:
            st.session_state.period_preset_key = "3Y"  # Default recommendation
        
        # Use session state as single source of truth
        default_period = st.session_state.period_preset_key
        
        try:
            default_period_index = list(period_options.keys()).index(default_period)
        except ValueError:
            default_period_index = 0  # Default to 3Y
            st.session_state.period_preset_key = "3Y"
        
        def on_period_change():
            """Callback to update session state when period changes."""
            st.session_state.period_preset_key = st.session_state.period_selector
        
        selected_period = st.selectbox(
            "Select analysis time period",
            options=list(period_options.keys()),
            index=default_period_index,
            format_func=lambda x: period_options[x],
            key="period_selector",
            help="Choose the time period for analysis. 3 years is recommended for balanced results.",
            on_change=on_period_change
        )
        
        # Show period description
        st.caption(period_descriptions.get(selected_period, ""))
        
        # Calculate dates or show custom inputs
        if selected_period == "CUSTOM":
            
            # Get current values as defaults
            try:
                current_start, current_end = data_access_service.get_date_range()
                default_start = current_start or datetime.now() - timedelta(days=365*3)
                default_end = current_end or datetime.now()
            except:
                default_start = datetime.now() - timedelta(days=365*3)
                default_end = datetime.now()
            
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
            
            date_range_start = datetime.combine(start_date, datetime.min.time()) if start_date else None
            date_range_end = datetime.combine(end_date, datetime.min.time()) if end_date else None
            
            # Validate custom range
            if date_range_start and date_range_end:
                if date_range_start >= date_range_end:
                    st.error("Start date must be before end date")
                elif (date_range_end - date_range_start).days < 30:
                    st.warning("Very short time period may not provide reliable results")
                else:
                    days_span = (date_range_end - date_range_start).days
                    st.success(f"Custom range: {days_span} days ({days_span/365:.1f} years)")
        else:
            # Use preset dates
            date_range_start, date_range_end = get_date_range(selected_period)
            
            # Display the calculated range
            days_span = (date_range_end - date_range_start).days
            st.success(f"Period: {date_range_start.strftime('%Y-%m-%d')} to {date_range_end.strftime('%Y-%m-%d')} ({days_span/365:.1f} years)")
        
        # Show data adequacy warning for high frequencies with short periods
        if selected_frequency in ["Q", "A"] and selected_period in ["1Y"]:
            if selected_frequency == "Q":
                st.warning("Quarterly frequency with 1-year period provides only ~4 data points")
            else:
                st.warning("Annual frequency with 1-year period provides only 1 data point")
        
    
    return SidebarState(
        lens=lens,
        selected_component_id=selected_component_id,
        selected_risk_model=selected_risk_model,
        selected_portfolio_graph=selected_portfolio_graph,
        selected_factors=[],  # Simplified - no factor filtering for now
        annualized=True,  # Always annualized for consistency 
        show_percentage=True,  # Always show percentages
        frequency=selected_frequency,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        date_range_preset=selected_period
    )