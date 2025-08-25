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
    from .components.tabs.correlations import render_correlations_tab
    from .components.tabs.data_management import render_data_management_tab
except ImportError:
    # Fall back to absolute imports (when run standalone)
    from data_loader import DataLoader
    from components.sidebar import render_sidebar
    from components.tabs.overview import render_overview_tab
    from components.tabs.active_lens import render_active_lens_tab
    from components.tabs.correlations import render_correlations_tab
    from components.tabs.data_management import render_data_management_tab

def run():
    """Entry point for integration with main Spark UI launcher."""
    # Initialize data loader
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    data_loader = st.session_state.data_loader
    
    # Render sidebar and get filter states
    sidebar_state = render_sidebar(data_loader)
    
    # Main header
    st.title("Maverick Risk Analysis")
    
    # Breadcrumb and badges (placeholder for now)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Portfolio Path:** TOTAL")
    with col2:
        st.success("✓ Validated")
        st.info("USD")
    
    # Tab navigation
    tab_names = [
        "Overview",
        "Active Lens", 
        "Hierarchy Explorer",
        "Timeline",
        "Factor Lens",
        "Assets",
        "Weights & Tilts",
        "Decomposition",
        "Stats & Distributions", 
        "Correlations",
        "Validation",
        "Data Browser"
    ]
    
    tabs = st.tabs(tab_names)
    
    # Tab 1 - Overview
    with tabs[0]:
        render_overview_tab(data_loader, sidebar_state)
    
    # Tab 2 - Active Lens
    with tabs[1]:
        render_active_lens_tab(data_loader, sidebar_state)
    
    # Tab 10 - Correlations (with new scatter plot)
    with tabs[9]:
        render_correlations_tab(data_loader, sidebar_state)
    
    # Placeholder tabs for now
    for i, tab in enumerate(tabs[2:9]):
        with tab:
            st.info(f"Tab {i+3} - {tab_names[i+2]} - Coming soon")
    
    with tabs[10]:
        st.info("Tab 11 - Validation - Coming soon")
    
    with tabs[11]:
        render_data_management_tab(data_loader, sidebar_state)

def main():
    """Standalone entry point when running maverick directly."""
    st.set_page_config(
        page_title="Maverick - Risk Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    run()

if __name__ == "__main__":
    main()