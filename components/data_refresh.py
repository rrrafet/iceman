"""
Data refresh UI components for Maverick application.

Provides UI controls for refreshing risk analysis data and managing cache.
"""

import streamlit as st
from typing import Dict, Any
import os

def render_data_refresh_controls(data_loader) -> bool:
    """
    Render data refresh controls in the sidebar or main area.
    
    Args:
        data_loader: DataLoader instance
        
    Returns:
        True if data was refreshed, False otherwise
    """
    data_refreshed = False
    
    st.subheader("🔄 Data Management")
    
    # Show data source status
    if data_loader.has_risk_service():
        st.success("✅ RiskAnalysisService available")
        
        # Refresh button
        if st.button("🔄 Refresh Risk Data", key="refresh_risk_data"):
            with st.spinner("Refreshing risk analysis data..."):
                try:
                    data_loader.refresh_data("TOTAL")
                    st.success("✅ Data refreshed successfully!")
                    data_refreshed = True
                    # Trigger app rerun to update UI
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error refreshing data: {e}")
        
        # Cache management
        with st.expander("💾 Cache Management"):
            cache_info = data_loader.get_cache_info()
            
            st.write("**Cache Directory:**")
            st.code(cache_info.get('cache_dir', 'N/A'))
            
            cache_files = cache_info.get('cache_files', [])
            if cache_files:
                st.write(f"**Cached Files:** {len(cache_files)}")
                
                for file_info in cache_files:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.text(file_info['filename'])
                    with col2:
                        st.text(f"{file_info['size_mb']:.2f} MB")
                    with col3:
                        st.text(file_info['modified'][:10])  # Show date only
            else:
                st.info("No cached files found")
        
        # Upload schema option
        with st.expander("📤 Upload Schema from Notebook"):
            st.markdown("""
            **To upload a schema from your Jupyter notebook:**
            
            1. In your notebook, run:
            ```python
            schema = analyzer.get_comprehensive_schema("TOTAL", factor_returns=factor_returns)
            data_loader.save_schema_to_cache(schema, "TOTAL")
            ```
            
            2. Click refresh above to load the new data
            """)
            
            if st.button("📂 Open Cache Directory", key="open_cache_dir"):
                cache_dir = cache_info.get('cache_dir', '')
                if os.path.exists(cache_dir):
                    st.info(f"Cache directory: {cache_dir}")
                else:
                    st.warning("Cache directory not found")
    
    else:
        st.warning("⚠️ RiskAnalysisService not available")
        st.info("Using static data file or mock data")
        
        # Option to reinitialize with portfolio components
        with st.expander("🔧 Enable Dynamic Risk Analysis"):
            st.markdown("""
            **To enable dynamic risk analysis:**
            
            1. In your notebook, after creating portfolio_graph and factor_returns:
            ```python
            # Save components to cache for Streamlit to pick up
            import joblib
            joblib.dump({
                'portfolio_graph': graph,
                'factor_returns': factor_returns
            }, 'spark-ui/apps/maverick/cache/portfolio_components.pkl')
            ```
            
            2. Restart the Streamlit app to load components
            """)
    
    return data_refreshed

def render_data_source_info(data_loader) -> None:
    """
    Display information about the current data source.
    
    Args:
        data_loader: DataLoader instance
    """
    st.subheader("📊 Data Source Information")
    
    # Basic metadata
    metadata = data_loader.data.get('metadata', {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Analysis Type", metadata.get('analysis_type', 'N/A'))
        st.metric("Schema Version", metadata.get('schema_version', 'N/A'))
    
    with col2:
        st.metric("Data Frequency", metadata.get('data_frequency', 'N/A'))
        timestamp = metadata.get('timestamp', 'N/A')
        if timestamp != 'N/A' and len(timestamp) > 10:
            timestamp = timestamp[:10]  # Show date only
        st.metric("Last Updated", timestamp)
    
    # Data completeness check
    st.subheader("📋 Data Completeness")
    
    sections = ['portfolio', 'benchmark', 'active', 'time_series', 'weights']
    completeness = {}
    
    for section in sections:
        section_data = data_loader.data.get(section, {})
        if section_data:
            completeness[section] = "✅ Available"
        else:
            completeness[section] = "❌ Missing"
    
    # Display completeness
    cols = st.columns(len(sections))
    for i, (section, status) in enumerate(completeness.items()):
        with cols[i]:
            st.metric(section.title(), status)
    
    # Time series info
    time_series = data_loader.data.get('time_series', {})
    if time_series:
        ts_metadata = time_series.get('metadata', {})
        if ts_metadata:
            st.subheader("📈 Time Series Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Start Date", ts_metadata.get('start_date', 'N/A'))
            with col2:
                st.metric("End Date", ts_metadata.get('end_date', 'N/A'))
            with col3:
                # Count periods from sample data
                portfolio_returns = time_series.get('portfolio_returns', {})
                if portfolio_returns:
                    sample_data = list(portfolio_returns.values())[0]
                    periods = len(sample_data) if isinstance(sample_data, list) else 'N/A'
                    st.metric("Periods", periods)
                else:
                    st.metric("Periods", "N/A")

def render_quick_save_helper() -> None:
    """
    Display helper for quickly saving schema from notebook.
    """
    with st.expander("💡 Quick Save Helper"):
        st.markdown("""
        **Copy and paste this code in your notebook to save current analysis:**
        
        ```python
        # After running your risk analysis
        schema = analyzer.get_comprehensive_schema("TOTAL", factor_returns=factor_returns)
        
        # Save to Maverick cache
        import sys
        import os
        sys.path.append('/Users/rafet/Workspace/Spark/spark-ui/apps/maverick')
        
        from services.risk_service import RiskAnalysisService
        service = RiskAnalysisService()
        cache_path = service.save_schema_to_cache(schema, "TOTAL")
        print(f"Schema saved to: {cache_path}")
        
        # Alternative: Save data dict directly
        import pickle
        with open('/Users/rafet/Workspace/Spark/spark-ui/apps/maverick/cache/risk_data_TOTAL.pkl', 'wb') as f:
            pickle.dump(schema.data, f)
        print("Data saved to cache!")
        ```
        """)