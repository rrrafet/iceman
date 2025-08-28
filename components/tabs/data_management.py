"""
Data management tab for Maverick application.

Provides comprehensive data management interface including refresh controls,
cache management, and integration helpers.
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from components.data_refresh import render_data_source_info, render_quick_save_helper

def render_data_management_tab(data_loader, sidebar_state):
    """Render Tab 12 - Data Management & Integration"""
    
    st.header("Data Management & Integration")
    st.markdown("Manage risk analysis data sources and integration with Jupyter notebooks")
    
    # Two columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Data source information
        render_data_source_info(data_loader)
        
        st.divider()
        
        # Integration status
        st.subheader("Integration Status")
        
        if data_loader.has_risk_service():
            st.success("RiskAnalysisService is active")
            
            # Show integration capabilities
            st.markdown("""
            **Active Capabilities:**
            - Real-time risk data generation
            - Cache management
            - Schema serialization
            - Dynamic data refresh
            """)
            
            # Cache statistics
            cache_info = data_loader.get_cache_info()
            cache_files = cache_info.get('cache_files', [])
            
            if cache_files:
                st.subheader("Cache Statistics")
                
                total_size = sum(f['size_mb'] for f in cache_files)
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Cached Files", len(cache_files))
                with col_stats2:
                    st.metric("Total Size", f"{total_size:.2f} MB")
                with col_stats3:
                    latest_file = max(cache_files, key=lambda x: x['modified'])
                    st.metric("Latest Update", latest_file['modified'][:10])
                
                # File details table
                st.subheader("Cache Files")
                import pandas as pd
                cache_df = pd.DataFrame(cache_files)
                cache_df['size_mb'] = cache_df['size_mb'].round(2)
                cache_df['modified'] = cache_df['modified'].str[:19]  # Remove microseconds
                st.dataframe(cache_df, use_container_width=True)
        
        else:
            st.warning("RiskAnalysisService not available")
            st.markdown("""
            **Current Limitations:**
            - No real-time data generation
            - No cache management
            - Static data source only
            
            **To enable full integration:**
            1. Ensure Spark modules are available in Python path
            2. Provide portfolio_graph and factor_returns to DataLoader
            3. Restart the Streamlit application
            """)
    
    with col2:
        # Quick save helper
        render_quick_save_helper()
        
        st.divider()
        
        # Technical information
        st.subheader("Technical Info")
        
        # Python path info
        with st.expander("Python Path"):
            for i, path in enumerate(sys.path[:10]):  # Show first 10 paths
                st.code(f"{i}: {path}")
            if len(sys.path) > 10:
                st.text(f"... and {len(sys.path) - 10} more paths")
        
        # Environment info
        with st.expander("Environment"):
            st.code(f"Working Directory: {os.getcwd()}")
            st.code(f"Script Directory: {os.path.dirname(__file__)}")
            
            # Check for key modules
            modules_to_check = [
                'spark.portfolio.risk_analyzer',
                'spark.portfolio.graph', 
                'spark.risk.schema',
                'joblib',
                'pickle'
            ]
            
            st.markdown("**Module Availability:**")
            for module_name in modules_to_check:
                try:
                    __import__(module_name)
                    st.success(f"Available: {module_name}")
                except ImportError:
                    st.error(f"Missing: {module_name}")
    
    st.divider()
    
    # Workflow examples
    st.subheader("Integration Workflows")
    
    # Tabs for different workflows
    workflow_tabs = st.tabs([
        "Notebook → Streamlit",
        "Real-time Integration", 
        "Cache Management",
        "Troubleshooting"
    ])
    
    with workflow_tabs[0]:
        st.markdown("""
        ### Jupyter Notebook → Streamlit Workflow
        
        **Step 1: Generate Risk Analysis in Notebook**
        ```python
        # Your existing notebook code
        analyzer = PortfolioRiskAnalyzer(graph)
        visitor = analyzer.decompose_factor_risk("TOTAL", factor_returns=factor_returns)
        schema = analyzer.get_riskresult("TOTAL", factor_returns=factor_returns, include_time_series=True)
        ```
        
        **Step 2: Save to Maverick Cache**
        ```python
        # Option A: Save schema object (recommended)
        import sys
        sys.path.append('/Users/rafet/Workspace/Spark/spark-ui/apps/maverick')
        from services.risk_service import RiskAnalysisService
        
        service = RiskAnalysisService()
        cache_path = service.save_schema_to_cache(schema, "TOTAL")
        print(f"Saved to: {cache_path}")
        
        # Option B: Save data dictionary directly
        import pickle
        cache_dir = '/Users/rafet/Workspace/Spark/spark-ui/apps/maverick/cache'
        with open(f'{cache_dir}/risk_data_TOTAL.pkl', 'wb') as f:
            pickle.dump(schema.data, f)
        ```
        
        **Step 3: Refresh Streamlit**
        - Click the "Refresh Risk Data" button in the sidebar
        - Or restart the Streamlit app to pick up new cache files
        """)
    
    with workflow_tabs[1]:
        st.markdown("""
        ### Real-time Integration Setup
        
        **For Direct Integration (Advanced)**
        ```python
        # In your notebook, create a service instance
        from services.risk_service import RiskAnalysisService
        
        # Initialize with your components
        service = RiskAnalysisService(
            portfolio_graph=graph,
            factor_returns=factor_returns
        )
        
        # Test data generation
        data = service.get_risk_data("TOTAL")
        print("Real-time data generated successfully!")
        ```
        
        **For Streamlit Integration**
        ```python
        # Initialize DataLoader with components
        data_loader = DataLoader(
            portfolio_graph=graph,
            factor_returns=factor_returns,
            use_risk_service=True
        )
        ```
        """)
    
    with workflow_tabs[2]:
        st.markdown("""
        ### Cache Management
        
        **View Cache Files**
        ```python
        service = RiskAnalysisService()
        info = service.get_cache_info()
        print(f"Cache directory: {info['cache_dir']}")
        for file in info['cache_files']:
            print(f"- {file['filename']} ({file['size_mb']:.2f} MB)")
        ```
        
        **Clear Cache**
        ```python
        import os
        import glob
        
        cache_dir = '/Users/rafet/Workspace/Spark/spark-ui/apps/maverick/cache'
        cache_files = glob.glob(f'{cache_dir}/*.pkl')
        
        for file_path in cache_files:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        ```
        
        **Manual Cache Load**
        ```python
        import pickle
        
        cache_path = '/Users/rafet/Workspace/Spark/spark-ui/apps/maverick/cache/risk_data_TOTAL.pkl'
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        print("Data loaded from cache:", list(data.keys()))
        ```
        """)
    
    with workflow_tabs[3]:
        st.markdown("""
        ### Troubleshooting Common Issues
        
        **Issue: "RiskAnalysisService not available"**
        - Check that Spark modules are in Python path
        - Ensure portfolio_graph and factor_returns are provided
        - Verify that required dependencies are installed
        
        **Issue: "Cache files not loading"**
        - Check cache directory permissions
        - Verify pickle files are not corrupted
        - Try clearing cache and regenerating
        
        **Issue: "Data refresh not working"**
        - Ensure portfolio components are properly initialized
        - Check for errors in the console/logs
        - Try restarting the Streamlit app
        
        **Issue: "Import errors"**
        - Verify Spark package installation
        - Check Python path configuration
        - Ensure all required dependencies are available
        
        **Getting Help**
        - Check the console output for detailed error messages
        - Review the cache info in the sidebar
        - Use the environment information above to diagnose issues
        """)
    
    # Footer with system info
    st.divider()
    st.caption(f"Maverick Data Management • Running from: {os.path.dirname(__file__)}")