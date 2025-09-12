import streamlit as st
import os
from pathlib import Path

# Use absolute imports
from spark.ui.apps.maverick.services.configuration_service import ConfigurationService
from spark.ui.apps.maverick.services.risk_analysis_service import RiskAnalysisService
from spark.ui.apps.maverick.services.data_access_service import DataAccessService
from spark.ui.apps.maverick.components.sidebar import render_sidebar
from spark.ui.apps.maverick.components.tabs.overview import render_overview_tab
from spark.ui.apps.maverick.components.tabs.risk_decomposition import render_risk_decomposition_tab
from spark.ui.apps.maverick.components.tabs.allocation_selection import render_allocation_selection_tab
from spark.ui.apps.maverick.components.tabs.data_explorer import render_data_explorer_tab
from spark.ui.apps.maverick.components.tabs.reconciliation import render_reconciliation_tab
from spark.ui.apps.maverick.datamodels import FactorDataProvider, PortfolioDataProvider

def initialize_services():
    """Initialize the 3-layer architecture services."""
    try:
        # Determine config path - check multiple locations
        config_locations = [
            # Environment variable
            os.environ.get('MAVERICK_CONFIG_PATH'),
            # Local config directory
            'spark-ui/spark/ui/apps/maverick/config/default_config.yaml',
            # Relative to current directory
            'config/default_config.yaml',
            # Package default location
            Path(__file__).parent / 'config' / 'default_config.yaml'
        ]
        
        config_path = None
        for location in config_locations:
            if location and Path(location).exists():
                config_path = location
                break
        
        if not config_path:
            raise FileNotFoundError("Could not find configuration file in any expected location")
        
        # Initialize configuration service
        config_service = ConfigurationService(config_path)
        
        # Get resolved data file paths using configuration service
        factor_data_path = config_service.get_data_source_path('factor_returns')
        portfolio_config_path = config_service.get_data_source_path('portfolio_config')
        
        # Initialize data providers with resolved paths
        factor_provider = FactorDataProvider(str(factor_data_path))
        portfolio_provider = PortfolioDataProvider(str(portfolio_config_path))
        
        # Initialize risk analysis service
        risk_analysis_service = RiskAnalysisService(
            config_service=config_service,
            factor_provider=factor_provider,
            portfolio_provider=portfolio_provider
        )
        
        # Initialize the risk analysis service (builds portfolio graph, etc.)
        if not risk_analysis_service.initialize():
            raise RuntimeError("Failed to initialize risk analysis service")
        
        # Initialize data access service
        data_access_service = DataAccessService(risk_analysis_service)
        
        return config_service, risk_analysis_service, data_access_service
    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'st' in globals():
            st.error(f"Failed to initialize services: {e}")
        return None, None, None

def run():
    """Entry point for integration with main Spark UI launcher."""
    # Initialize services on first run
    if 'services_initialized' not in st.session_state:
        with st.spinner("Loading portfolio and risk models..."):
            config_service, risk_analysis_service, data_access_service = initialize_services()
            
            if config_service is None:
                st.error("Failed to initialize Maverick. Please check configuration.")
                return
            
            # Store in session state
            st.session_state.config_service = config_service
            st.session_state.risk_analysis_service = risk_analysis_service 
            st.session_state.data_access_service = data_access_service
            st.session_state.services_initialized = True
            
            #st.success(f"Loaded portfolio: {config_service.get_portfolio_name()}")
            #st.success(f"Loaded risk model: {config_service.get_default_risk_model()}")
    
    # Get services from session state
    config_service = st.session_state.config_service
    risk_analysis_service = st.session_state.risk_analysis_service
    data_access_service = st.session_state.data_access_service
    
    # Render sidebar and get filter states
    sidebar_state = render_sidebar(config_service, data_access_service)
    
    # Handle portfolio graph changes
    current_portfolio_graph = config_service.get_default_portfolio_graph()
    if sidebar_state.selected_portfolio_graph != current_portfolio_graph:
        with st.spinner(f"Loading portfolio graph: {sidebar_state.selected_portfolio_graph}..."):
            try:
                # Get the new portfolio config path
                new_portfolio_config_path = config_service.get_portfolio_graph_path(sidebar_state.selected_portfolio_graph)
                
                # Reinitialize services with new portfolio
                factor_data_path = config_service.get_data_source_path('factor_returns')
                factor_provider = FactorDataProvider(str(factor_data_path))
                portfolio_provider = PortfolioDataProvider(str(new_portfolio_config_path))
                
                # Create new risk analysis service
                new_risk_analysis_service = RiskAnalysisService(
                    config_service=config_service,
                    factor_provider=factor_provider,
                    portfolio_provider=portfolio_provider
                )
                
                # Initialize the new risk analysis service
                if new_risk_analysis_service.initialize():
                    # Create new data access service
                    new_data_access_service = DataAccessService(new_risk_analysis_service)
                    
                    # Update session state
                    st.session_state.risk_analysis_service = new_risk_analysis_service
                    st.session_state.data_access_service = new_data_access_service
                    
                    # Update config service to reflect new default
                    config_service.update_setting("portfolio_graphs.default", sidebar_state.selected_portfolio_graph)
                    
                    st.success(f"Portfolio graph changed to {sidebar_state.selected_portfolio_graph}")
                    st.info("🔄 Portfolio and risk calculations re-initialized with new graph structure.")
                    # Force a rerun to refresh all data with new portfolio
                    st.rerun()
                else:
                    st.error(f"Failed to initialize risk analysis with portfolio: {sidebar_state.selected_portfolio_graph}")
            except Exception as e:
                st.error(f"Failed to change portfolio graph: {e}")
    
    # Handle risk model changes
    current_model = data_access_service.get_current_risk_model()
    if sidebar_state.selected_risk_model != current_model:
        with st.spinner(f"Loading risk model: {sidebar_state.selected_risk_model}..."):
            try:
                success = data_access_service.switch_risk_model(sidebar_state.selected_risk_model)
                if success:
                    st.success(f"Risk model changed to {sidebar_state.selected_risk_model}")
                    st.info("🔄 Risk calculations re-initialized with new factor model.")
                    # Force a rerun to refresh all data with new model
                    st.rerun()
                else:
                    st.error(f"Failed to switch to risk model: {sidebar_state.selected_risk_model}")
            except Exception as e:
                st.error(f"Failed to change risk model: {e}")
    
    # Handle frequency changes
    current_freq = data_access_service.get_current_frequency()
    if sidebar_state.frequency != current_freq:
        with st.spinner(f"Switching to {sidebar_state.frequency} frequency..."):
            try:
                success = data_access_service.set_frequency(sidebar_state.frequency)
                if success:
                    st.success(f"Data frequency changed to {sidebar_state.frequency}")
                    st.info("🔄 System re-initialized with new frequency. All charts and tables now show resampled data.")
                    # Force a rerun to refresh all data with new frequency
                    st.rerun()
                else:
                    st.warning("Frequency was already set to the selected value")
            except Exception as e:
                st.error(f"Failed to change frequency: {e}")
    
    # Handle date range changes
    current_start, current_end = data_access_service.get_date_range()
    if sidebar_state.date_range_start != current_start or sidebar_state.date_range_end != current_end:
        with st.spinner(f"Applying date range filter..."):
            try:
                success = data_access_service.set_date_range(sidebar_state.date_range_start, sidebar_state.date_range_end)
                if success:
                    start_str = sidebar_state.date_range_start.strftime('%Y-%m-%d') if sidebar_state.date_range_start else 'None'
                    end_str = sidebar_state.date_range_end.strftime('%Y-%m-%d') if sidebar_state.date_range_end else 'None'
                    st.success(f"Date range applied: {start_str} to {end_str}")
                    st.info("🔄 System re-initialized with new date range. All data filtered to selected period.")
                    # Force a rerun to refresh all data with new date range
                    st.rerun()
                else:
                    st.warning("Date range was already set to the selected values")
            except Exception as e:
                st.error(f"Failed to change date range: {e}")
    
    # Debug: Show current state (can be removed later)
    if st.sidebar.checkbox("Show Debug Info", value=False):
        st.sidebar.write("**Debug Info:**")
        st.sidebar.write(f"Sidebar date range: {sidebar_state.date_range_start} to {sidebar_state.date_range_end}")
        st.sidebar.write(f"Data service date range: {current_start} to {current_end}")
        st.sidebar.write(f"Preset: {sidebar_state.date_range_preset}")
        
        # Show data provider info to verify single source of truth
        try:
            # Portfolio data info
            portfolio_provider = data_access_service.portfolio_provider
            if portfolio_provider and hasattr(portfolio_provider, '_data') and portfolio_provider._data is not None:
                st.sidebar.write(f"**Portfolio Data Source:**")
                st.sidebar.write(f"Total records: {len(portfolio_provider._data)}")
                if not portfolio_provider._data.empty:
                    min_date = portfolio_provider._data['date'].min().strftime('%Y-%m-%d')
                    max_date = portfolio_provider._data['date'].max().strftime('%Y-%m-%d')
                    st.sidebar.write(f"Date range: {min_date} to {max_date}")
                    components = portfolio_provider._data['component_id'].nunique()
                    st.sidebar.write(f"Components: {components}")
            
            # Factor data info  
            factor_provider = data_access_service.factor_provider
            if factor_provider and hasattr(factor_provider, '_data') and factor_provider._data is not None:
                st.sidebar.write(f"**Factor Data Source:**")
                st.sidebar.write(f"Total records: {len(factor_provider._data)}")
                if not factor_provider._data.empty:
                    min_date = factor_provider._data['date'].min().strftime('%Y-%m-%d')
                    max_date = factor_provider._data['date'].max().strftime('%Y-%m-%d')
                    st.sidebar.write(f"Date range: {min_date} to {max_date}")
                    factors = factor_provider._data['factor_name'].nunique()
                    st.sidebar.write(f"Factors: {factors}")
                    
            # Verification: Get sample returns to confirm filtering is applied
            sample_component = data_access_service.get_all_component_ids()[0] if data_access_service.get_all_component_ids() else None
            if sample_component:
                sample_returns = data_access_service.get_portfolio_returns(sample_component)
                st.sidebar.write(f"**Processed Returns:**")
                st.sidebar.write(f"Return series length: {len(sample_returns)}")
                if not sample_returns.empty:
                    st.sidebar.write(f"Series range: {sample_returns.index.min().strftime('%Y-%m-%d')} to {sample_returns.index.max().strftime('%Y-%m-%d')}")
                    
        except Exception as e:
            st.sidebar.write(f"Debug error: {e}")
    
    # Main header
    st.title("Maverick")
    
    # Enhanced header with portfolio and risk analysis status
    col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1.5])
    
    with col1:
        # Portfolio info
        portfolio_name = config_service.get_portfolio_name()
        st.markdown(f"**Graph:** {portfolio_name}")
        
        # Show current component path
        portfolio_graph = data_access_service.risk_analysis_service.get_portfolio_graph()
        selected_component = sidebar_state.selected_component_id if hasattr(sidebar_state, 'selected_component_id') else config_service.get_root_component_id(portfolio_graph)
        #st.caption(f"Component: {selected_component}")
    
    with col2:
        # Risk model and analysis status  
        current_risk_model = data_access_service.get_current_risk_model()
        st.markdown(f"**Risk Model:** {current_risk_model}")
        #st.caption("Risk Analysis: Ready")
    
    with col3:
        lens = sidebar_state.lens if hasattr(sidebar_state, 'lens') else config_service.get_default_lens()
        st.markdown(f"**Lens:** {lens.title()}")
        #currency = config_service.get_currency()
        #st.info(currency)
    
    with col4:
        # Show current frequency
        frequency_labels = {
            "D": "Daily", 
            "B": "Business Daily",
            "W-FRI": "Weekly", 
            "ME": "Monthly"
        }
        freq_label = frequency_labels.get(sidebar_state.frequency, sidebar_state.frequency)
        st.markdown(f"**Freq:** {freq_label}")
        if sidebar_state.frequency not in ["D", "B"]:
            st.caption("📈 Resampled")
    
    with col5:
        # Show current date range
        if sidebar_state.date_range_start and sidebar_state.date_range_end:
            start_str = sidebar_state.date_range_start.strftime('%m/%d/%y')
            end_str = sidebar_state.date_range_end.strftime('%m/%d/%y')
            st.markdown(f"**Date Range:** {sidebar_state.date_range_preset}")
            st.caption(f"📅 {start_str} - {end_str}")
        else:
            st.markdown("**Date Range:** All Data")
            st.caption("📅 No filter")
    
    # Tab navigation with state persistence
    tab_names = [
        "Overview",
        "Risk Decomposition",
        "Allocation-Selection", 
        "Data Explorer",
        "Reconciliation",
    ]
    
    # Initialize active tab in session state if not exists
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Overview"
    
    # Initialize the radio widget state to match session state
    if 'tab_radio' not in st.session_state:
        st.session_state.tab_radio = st.session_state.active_tab
    
    # Define callback function to update active tab
    def on_tab_change():
        st.session_state.active_tab = st.session_state.tab_radio
    
    # Create tab selector with callback
    selected_tab = st.radio(
        "Navigation",
        options=tab_names,
        key="tab_radio",
        horizontal=True,
        label_visibility="collapsed",
        on_change=on_tab_change
    )
    
    # Use the selected tab value
    current_tab = selected_tab
    
    # Add some visual separation
    st.divider()
    
    # Render the selected tab content
    if current_tab == "Overview":
        render_overview_tab(data_access_service, sidebar_state)
    elif current_tab == "Risk Decomposition":
        render_risk_decomposition_tab(data_access_service, sidebar_state)
    elif current_tab == "Allocation-Selection":
        render_allocation_selection_tab(data_access_service, sidebar_state)
    elif current_tab == "Data Explorer":
        render_data_explorer_tab(data_access_service, sidebar_state)
    elif current_tab == "Reconciliation":
        render_reconciliation_tab(data_access_service, sidebar_state)


def main():
    """Standalone entry point when running maverick directly."""
    st.set_page_config(
        page_title="Maverick - Risk Analysis",
        page_icon="M",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    run()

if __name__ == "__main__":
    main()