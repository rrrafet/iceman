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
    
    # Main header
    st.title("Maverick")
    
    # Enhanced header with portfolio and risk analysis status
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        # Portfolio info
        portfolio_name = config_service.get_portfolio_name()
        st.markdown(f"**Graph:** {portfolio_name}")
        
        # Show current component path
        selected_component = sidebar_state.selected_component_id if hasattr(sidebar_state, 'selected_component_id') else config_service.get_root_component_id()
        #st.caption(f"Component: {selected_component}")
    
    with col2:
        # Risk model and analysis status  
        current_risk_model = sidebar_state.selected_risk_model if hasattr(sidebar_state, 'selected_risk_model') else config_service.get_default_risk_model()
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
    
    # Tab navigation
    tab_names = [
        "Overview",
        "Risk Decomposition",
        "Allocation-Selection",
        "Data Explorer",
        # "Active Lens", 
        # "Hierarchy Explorer",
        # "Timeline",
        # "Factor Lens",
        # "Assets",
        # "Weights & Tilts",
        # "Decomposition",
        # "Stats & Distributions", 
        # "Correlations",
        # "Validation & Diagnostics",
        # "Data Management & Integration"
    ]
    
    tabs = st.tabs(tab_names)
    
    # Tab 1 - Overview
    with tabs[0]:
        render_overview_tab(data_access_service, sidebar_state)
    
    # Tab 2 - Risk Decomposition
    with tabs[1]:
        render_risk_decomposition_tab(data_access_service, sidebar_state)
    
    # Tab 3 - Allocation-Selection
    with tabs[2]:
        render_allocation_selection_tab(data_access_service, sidebar_state)
    
    # Tab 4 - Data Explorer
    with tabs[3]:
        render_data_explorer_tab(data_access_service, sidebar_state)


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