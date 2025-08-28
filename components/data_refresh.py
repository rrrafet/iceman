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
        st.success("✅ Risk Analysis Service available")
        
        # Show current status
        try:
            risk_status = data_loader.get_risk_analysis_status()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Portfolio Components", risk_status.get('portfolio_components', 0))
            with col2:
                st.metric("Risk Factors", risk_status.get('factor_count', 0))
                
            if risk_status.get('analysis_completed', False):
                st.info("✅ Risk analysis data available")
            elif risk_status.get('ready_for_analysis', False):
                st.warning("⏳ Ready to run analysis")
            else:
                st.warning("⚠️ Waiting for portfolio & risk model")
                
        except Exception as e:
            st.error(f"Error getting risk status: {e}")
        
        # Refresh button
        if st.button("🔄 Refresh Configuration", key="refresh_config_data"):
            with st.spinner("Refreshing configuration data..."):
                try:
                    data_loader.refresh_data()
                    st.success("✅ Configuration refreshed!")
                    data_refreshed = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error refreshing: {e}")
        
        # Clear cache button
        if st.button("🗑️ Clear Risk Analysis Cache", key="clear_risk_cache"):
            try:
                if data_loader.risk_service:
                    data_loader.risk_service.clear_cache()
                    st.success("✅ Cache cleared!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing cache: {e}")
        
        # Cache info
        with st.expander("💾 Generated Data Files"):
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'config', 'data')
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
                if files:
                    st.write(f"**Generated Files:** {len(files)}")
                    for filename in files:
                        filepath = os.path.join(data_dir, filename)
                        if os.path.exists(filepath):
                            size_mb = os.path.getsize(filepath) / (1024*1024)
                            st.text(f"{filename} ({size_mb:.2f} MB)")
                else:
                    st.info("No data files found")
            else:
                st.warning("Data directory not found")
        
        # Integration info
        with st.expander("📤 Integration with Notebooks"):
            st.markdown("""
            **To integrate with Jupyter notebooks:**
            
            1. Generate portfolio data:
            ```python
            cd config
            python data_generator.py
            ```
            
            2. Load portfolio in notebook:
            ```python
            from config.portfolio_loader import load_portfolio_from_config_name
            result = load_portfolio_from_config_name('strategic_portfolio', 'config')
            portfolio_graph = result['portfolio_graph']
            factor_returns = result['factor_returns']
            ```
            
            3. Run risk analysis:
            ```python
            from spark.portfolio.risk_analyzer import PortfolioRiskAnalyzer
            analyzer = PortfolioRiskAnalyzer(portfolio_graph)
            schema = analyzer.get_riskresult('TOTAL', factor_returns)
            ```
            """)
            
            config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
            if st.button("📂 Show Config Directory", key="show_config_dir"):
                st.info(f"Configuration directory: {config_dir}")
    
    else:
        st.warning("⚠️ Risk Analysis Service not available")
        st.info("Check system configuration")
        
        # Option to reinitialize
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **If risk analysis is not working:**
            
            1. Check that all dependencies are installed
            2. Verify portfolio configuration is loaded
            3. Verify risk model is selected
            4. Try restarting the Streamlit app
            
            **Data files should be in:**
            - `config/data/portfolio.parquet`
            - `config/data/factor_returns_*.parquet`
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
        schema = analyzer.get_riskresult("TOTAL", factor_returns=factor_returns, include_time_series=True)
        
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