import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Handle imports for both standalone and module execution
try:
    # Try relative imports first (when used as module)
    from .services.configuration_service import ConfigurationService
    from .services.risk_analysis_service import RiskAnalysisService
    from .services.data_access_service import DataAccessService
    from .components.sidebar import render_sidebar
    from .components.tabs.overview import render_overview_tab
    from .components.tabs.risk_decomposition import render_risk_decomposition_tab
    from .components.tabs.allocation_selection import render_allocation_selection_tab
    from .components.tabs.data_explorer import render_data_explorer_tab
except ImportError:
    # Fall back to absolute imports (when run standalone)
    from services.configuration_service import ConfigurationService
    from services.risk_analysis_service import RiskAnalysisService
    from services.data_access_service import DataAccessService
    from components.sidebar import render_sidebar
    from components.tabs.overview import render_overview_tab
    from components.tabs.risk_decomposition import render_risk_decomposition_tab
    from components.tabs.allocation_selection import render_allocation_selection_tab
    from components.tabs.data_explorer import render_data_explorer_tab

def initialize_services():
    """Initialize the 3-layer architecture services."""
    try:
        # Get config path
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'default_config.yaml')
        
        # Initialize configuration service
        config_service = ConfigurationService(config_path)
        
        # Initialize data providers
        data_sources = config_service.get_data_sources()
        
        # Import data providers  
        from datamodels import FactorDataProvider, PortfolioDataProvider
        
        factor_provider = FactorDataProvider(
            os.path.join(os.path.dirname(__file__), data_sources.get('factor_returns', 'data/factor_returns.parquet'))
        )
        
        portfolio_provider = PortfolioDataProvider(
            os.path.join(os.path.dirname(__file__), data_sources.get('portfolio_config', 'graphs/strategic_portfolio.yaml'))
        )
        
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
    
    # Main header
    st.title("Maverick")
    
    # Enhanced header with portfolio and risk analysis status
    col1, col2, col3 = st.columns([2, 2, 1])
    
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