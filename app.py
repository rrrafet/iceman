import streamlit as st
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Handle imports for both standalone and module execution
try:
    # Try relative imports first (when used as module)
    from .data_loader import DataLoader
    from .components.sidebar import render_sidebar
    from .components.tabs.overview import render_overview_tab
    from .components.tabs.active_lens import render_active_lens_tab
    from .components.tabs.hierarchy_explorer import render_hierarchy_explorer_tab
    from .components.tabs.timeline import render_timeline_tab
    from .components.tabs.factor_lens import render_factor_lens_tab
    from .components.tabs.assets import render_assets_tab
    from .components.tabs.weights_tilts import render_weights_tilts_tab
    from .components.tabs.decomposition import render_decomposition_tab
    from .components.tabs.stats_distributions import render_stats_distributions_tab
    from .components.tabs.correlations import render_correlations_tab
    from .components.tabs.validation import render_validation_tab
    from .components.tabs.data_management import render_data_management_tab
    from .components.tabs.risk_decomposition import render_risk_decomposition_tab
except ImportError:
    # Fall back to absolute imports (when run standalone)
    from data_loader import DataLoader
    from components.sidebar import render_sidebar
    from components.tabs.overview import render_overview_tab
    from components.tabs.active_lens import render_active_lens_tab
    from components.tabs.hierarchy_explorer import render_hierarchy_explorer_tab
    from components.tabs.timeline import render_timeline_tab
    from components.tabs.factor_lens import render_factor_lens_tab
    from components.tabs.assets import render_assets_tab
    from components.tabs.weights_tilts import render_weights_tilts_tab
    from components.tabs.decomposition import render_decomposition_tab
    from components.tabs.stats_distributions import render_stats_distributions_tab
    from components.tabs.correlations import render_correlations_tab
    from components.tabs.validation import render_validation_tab
    from components.tabs.data_management import render_data_management_tab
    from components.tabs.risk_decomposition import render_risk_decomposition_tab

def run():
    """Entry point for integration with main Spark UI launcher."""
    # Initialize data loader
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    data_loader = st.session_state.data_loader
    
    # Render sidebar and get filter states
    sidebar_state = render_sidebar(data_loader)
    
    # Main header
    st.title("Maverick")
    
    # Enhanced header with portfolio and risk analysis status
    col1, col2, col3 = st.columns([2, 2, 1])
    
    # with col1:
    #     # Portfolio info
    #     current_config = getattr(st.session_state, 'selected_portfolio_config', 'Default')
    #     #st.markdown(f"**Portfolio:** {current_config}")
        
    #     # Show current component path
    #     selected_node = sidebar_state.selected_node if hasattr(sidebar_state, 'selected_node') else "TOTAL"
    #     #st.caption(f"Path: {selected_node}")
    
    # with col2:
    #     # Risk model and analysis status
    #     current_risk_model = getattr(st.session_state, 'selected_risk_model', 'None')
    #     #st.markdown(f"**Risk Model:** {current_risk_model}")
        
    #     try:
    #         risk_status = data_loader.get_risk_analysis_status()
    #         if risk_status['analysis_completed']:
    #             st.caption("Risk Analysis: Complete")
    #         elif risk_status['ready_for_analysis']:
    #             st.caption("Risk Analysis: Ready")
    #         else:
    #             st.caption("Risk Analysis: Waiting")
    #     except:
    #         st.caption("Risk Analysis: Unknown")
    
    # with col3:
    #     #st.success("Validated")
    #     #st.info("USD")
    
    # Tab navigation
    tab_names = [
        "Overview",
        "Risk Decomposition",
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
        render_overview_tab(data_loader, sidebar_state)
    
    # Tab 2 - Risk Decomposition
    with tabs[1]:
        render_risk_decomposition_tab(data_loader, sidebar_state)
    
    # # Tab 3 - Active Lens
    # with tabs[1]:
    #     render_active_lens_tab(data_loader, sidebar_state)
    
    # # Tab 3 - Hierarchy Explorer
    # with tabs[2]:
    #     render_hierarchy_explorer_tab(data_loader, sidebar_state)
    
    # # Tab 4 - Timeline
    # with tabs[3]:
    #     render_timeline_tab(data_loader, sidebar_state)
    
    # # Tab 5 - Factor Lens
    # with tabs[4]:
    #     render_factor_lens_tab(data_loader, sidebar_state)
    
    # # Tab 6 - Assets
    # with tabs[5]:
    #     render_assets_tab(data_loader, sidebar_state)
    
    # # Tab 7 - Weights & Tilts
    # with tabs[6]:
    #     render_weights_tilts_tab(data_loader, sidebar_state)
    
    # # Tab 8 - Decomposition
    # with tabs[7]:
    #     render_decomposition_tab(data_loader, sidebar_state)
    
    # # Tab 9 - Stats & Distributions
    # with tabs[8]:
    #     render_stats_distributions_tab(data_loader, sidebar_state)
    
    # # Tab 10 - Correlations
    # with tabs[9]:
    #     render_correlations_tab(data_loader, sidebar_state)
    
    # # Tab 11 - Validation
    # with tabs[10]:
    #     render_validation_tab(data_loader, sidebar_state)
    
    # # Tab 12 - Data Management
    # with tabs[11]:
    #     render_data_management_tab(data_loader, sidebar_state)

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