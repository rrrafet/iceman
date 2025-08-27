import streamlit as st
from data_loader import SidebarState
from components.data_refresh import render_data_refresh_controls

def render_sidebar(data_loader) -> SidebarState:
    """Render global sidebar with filters and return state"""
    
    with st.sidebar:
        st.header("🎯 Maverick Controls")
        
        # Portfolio Configuration Selection
        st.subheader("📋 Portfolio Configuration")
        
        # Get available configurations
        try:
            import os
            from config.portfolio_loader import get_available_configurations
            
            config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
            available_configs = get_available_configurations(config_dir)
            
            if available_configs:
                # Create options for selectbox
                config_options = {f"{cfg['name']}\n{cfg['description'][:50]}..." if len(cfg['description']) > 50 else f"{cfg['name']}\n{cfg['description']}": cfg['id'] for cfg in available_configs}
                
                selected_config_display = st.selectbox(
                    "Select portfolio configuration",
                    options=list(config_options.keys()),
                    index=0,
                    key="portfolio_config_selector",
                    help="Choose a portfolio configuration to load"
                )
                
                selected_config_id = config_options[selected_config_display]
                
                # Show selected config details
                selected_config_info = next(cfg for cfg in available_configs if cfg['id'] == selected_config_id)
                st.caption(f"**{selected_config_info['name']}**")
                st.caption(selected_config_info['description'])
                
                # Button to load configuration
                if st.button("🔄 Load Configuration", key="load_config_btn"):
                    try:
                        # Store selected config in session state for data loader
                        st.session_state.selected_portfolio_config = selected_config_id
                        st.success(f"Loading {selected_config_info['name']}...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load configuration: {str(e)}")
            else:
                st.warning("No portfolio configurations found")
                st.caption("Add YAML files to config/portfolios/")
                
        except Exception as e:
            st.error("Portfolio configuration system unavailable")
            st.caption(f"Error: {str(e)}")
        
        st.divider()
        
        # Snapshot metadata display
        st.subheader("Snapshot")
        metadata = data_loader.data.get('metadata', {})
        st.text(f"Analysis: {metadata.get('analysis_type', 'N/A')}")
        st.text(f"Frequency: {metadata.get('data_frequency', 'N/A')}")
        st.text(f"Schema: v{metadata.get('schema_version', 'N/A')}")
        
        # Show current configuration if loaded
        if hasattr(st.session_state, 'selected_portfolio_config'):
            st.text(f"Config: {st.session_state.selected_portfolio_config}")
        
        st.divider()
        
        # Risk Model Selection
        st.subheader("🔬 Risk Model")
        
        try:
            from config.risk_model_loader import list_available_risk_models
            
            available_models = list_available_risk_models()
            
            if available_models:
                # Create dropdown options
                model_options = {}
                for model in available_models:
                    display_name = f"{model['name']}"
                    description = model['description'][:60] + "..." if len(model['description']) > 60 else model['description']
                    model_options[f"{display_name}\n{description}"] = model['id']
                
                selected_model_display = st.selectbox(
                    "Select risk model",
                    options=list(model_options.keys()),
                    index=0,
                    key="risk_model_selector",
                    help="Choose a risk model for factor analysis"
                )
                
                selected_model_id = model_options[selected_model_display]
                
                # Show selected model details
                selected_model = next(m for m in available_models if m['id'] == selected_model_id)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"**Factors:** {selected_model['num_factors']}")
                with col2:
                    st.caption(f"**Source:** {selected_model['source']}")
                
                st.caption(f"**Period:** {selected_model['start_date']} to {selected_model['end_date']}")
                
                # Show factor names (truncated)
                factors_display = ", ".join(selected_model['factors'][:5])
                if len(selected_model['factors']) > 5:
                    factors_display += f", ... (+{len(selected_model['factors']) - 5} more)"
                st.caption(f"**Factors:** {factors_display}")
                
                # Button to load risk model
                if st.button("🔄 Load Risk Model", key="load_risk_model_btn"):
                    try:
                        if data_loader.load_risk_model(selected_model_id):
                            st.session_state.selected_risk_model = selected_model_id
                            st.success(f"Loaded {selected_model['name']}")
                            st.rerun()
                        else:
                            st.error("Failed to load risk model")
                    except Exception as e:
                        st.error(f"Error loading risk model: {str(e)}")
                        
            else:
                st.warning("No risk models found")
                st.caption("Check risk model configuration")
                
        except Exception as e:
            st.error("Risk model system unavailable")
            st.caption(f"Error: {str(e)}")
        
        # Show current risk model if loaded
        if hasattr(st.session_state, 'selected_risk_model'):
            st.caption(f"**Active:** {st.session_state.selected_risk_model}")
        
        st.divider()
        
        # Risk Analysis Section  
        st.subheader("⚡ Risk Analysis")
        
        try:
            risk_status = data_loader.get_risk_analysis_status()
            
            if risk_status['available']:
                # Show current status
                if risk_status['ready_for_analysis']:
                    st.success("✅ Ready for analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Components: {risk_status['portfolio_components']}")
                    with col2:
                        st.caption(f"Factors: {risk_status['factor_count']}")
                    
                    # Run analysis button
                    if st.button("🔍 Run Risk Analysis", key="run_risk_analysis_btn"):
                        try:
                            success = data_loader.run_risk_analysis(force_refresh=True)
                            if success:
                                st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                    
                    # Show analysis status
                    if risk_status['analysis_completed']:
                        st.info(f"✅ Analysis completed")
                        if risk_status['last_analysis']:
                            # Parse timestamp for display
                            try:
                                from datetime import datetime
                                dt = datetime.fromisoformat(risk_status['last_analysis'].replace('Z', '+00:00'))
                                st.caption(f"Last run: {dt.strftime('%H:%M:%S')}")
                            except:
                                st.caption("Analysis available")
                    
                else:
                    st.warning("⚠️ Need portfolio & risk model")
                    if risk_status['portfolio_components'] == 0:
                        st.caption("→ Load portfolio configuration")
                    if risk_status['factor_count'] == 0:
                        st.caption("→ Load risk model")
                        
            else:
                st.error("Risk analysis unavailable")
                st.caption(risk_status.get('message', 'Unknown error'))
                
        except Exception as e:
            st.error("Risk analysis system error")
            st.caption(f"Error: {str(e)}")
        
        st.divider()
        
        # Lens selector
        st.subheader("📊 Lens")
        lens = st.selectbox(
            "Select view perspective",
            options=["portfolio", "benchmark", "active"],
            index=0,
            format_func=lambda x: x.title(),
            key="lens_selector"
        )
        
        # Enhanced hierarchical node selector
        st.subheader("🏗️ Hierarchical Navigator")
        
        # Get all available hierarchical components
        hierarchical_components = data_loader.get_available_hierarchical_components()
        all_component_names = data_loader.get_component_names()
        
        # Use hierarchical components if available, fallback to regular components
        available_components = hierarchical_components if hierarchical_components else all_component_names
        
        # Default to TOTAL if available
        default_node = "TOTAL" if "TOTAL" in available_components else (available_components[0] if available_components else "TOTAL")
        
        selected_node = st.selectbox(
            "Navigate hierarchy",
            options=available_components if available_components else ["TOTAL"],
            index=available_components.index(default_node) if default_node in available_components else 0,
            key="node_selector"
        )
        
        # Show hierarchy path breadcrumbs
        if selected_node and selected_node in hierarchical_components:
            hierarchy_path = data_loader.get_component_hierarchy_path(selected_node)
            if hierarchy_path and len(hierarchy_path) > 1:
                breadcrumb = " → ".join(hierarchy_path)
                st.caption(f"📍 Path: {breadcrumb}")
        
        # Show component metadata
        hierarchy = data_loader.get_hierarchy_info()
        component_metadata = hierarchy.get('component_metadata', {})
        if selected_node in component_metadata:
            metadata = component_metadata[selected_node]
            st.caption(f"Type: {metadata.get('type', 'N/A')} | Level: {metadata.get('level', 'N/A')}")
        
        # Navigation controls
        col1, col2 = st.columns(2)
        
        with col1:
            if data_loader.can_drill_up(selected_node):
                parent_id = data_loader.get_component_parent(selected_node)
                if st.button(f"⬆️ Up to {parent_id}", key="drill_up", help="Navigate to parent component"):
                    st.session_state.node_selector = parent_id
                    st.rerun()
        
        with col2:
            drill_down_options = data_loader.get_drilldown_options(selected_node)
            if drill_down_options:
                st.caption(f"⬇️ Can drill down to {len(drill_down_options)} child{'ren' if len(drill_down_options) > 1 else ''}")
        
        # Show drill-down options if available
        if drill_down_options:
            st.caption("**Children:**")
            drill_down_cols = st.columns(min(3, len(drill_down_options)))
            for i, child_id in enumerate(drill_down_options[:3]):  # Show first 3
                with drill_down_cols[i]:
                    if st.button(f"📂 {child_id}", key=f"drill_down_{child_id}", help=f"Navigate to {child_id}"):
                        st.session_state.node_selector = child_id
                        st.rerun()
            
            if len(drill_down_options) > 3:
                st.caption(f"... and {len(drill_down_options) - 3} more")
        
        # Show component-specific lens availability
        if selected_node in hierarchical_components:
            available_lenses = data_loader.get_component_lens_availability(selected_node)
            if available_lenses:
                st.caption(f"📊 Available lenses: {', '.join(available_lenses)}")
                
                # Validation indicator
                current_lens = lens  # from the lens selector above
                if current_lens in available_lenses:
                    validation = data_loader.get_component_validation_status(selected_node, current_lens)
                    if validation.get('euler_identity_check', False):
                        st.success("✅ Component validated")
                    else:
                        st.warning("⚠️ Validation issues")
        
        st.divider()
        
        # Date range selector
        st.subheader("📅 Date Range")
        time_series_meta = data_loader.data.get('time_series', {}).get('metadata', {})
        start_date = time_series_meta.get('start_date', '2023-01-01')
        end_date = time_series_meta.get('end_date', '2024-12-31')
        
        st.text(f"Available: {start_date} to {end_date}")
        
        # Get data length for slider
        sample_data = data_loader.get_time_series_data('portfolio_returns', 'TOTAL')
        max_periods = len(sample_data) if sample_data else 60
        
        date_range = st.slider(
            "Select period range",
            min_value=0,
            max_value=max_periods - 1,
            value=(0, max_periods - 1),
            key="date_range_slider"
        )
        
        # Preset buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("1Y", key="1y_preset"):
                st.session_state.date_range_slider = (max(0, max_periods - 252), max_periods - 1)
                st.rerun()
        with col2:
            if st.button("3Y", key="3y_preset"):
                st.session_state.date_range_slider = (max(0, max_periods - 756), max_periods - 1)
                st.rerun()
        with col3:
            if st.button("All", key="all_preset"):
                st.session_state.date_range_slider = (0, max_periods - 1)
                st.rerun()
        
        st.divider()
        
        # Factor filter
        st.subheader("🎛️ Factor Filter")
        factor_names = data_loader.get_factor_names()
        
        selected_factors = st.multiselect(
            "Select factors to analyze",
            options=factor_names,
            default=[],
            key="factor_filter"
        )
        
        if st.button("Select All Factors", key="select_all_factors"):
            st.session_state.factor_filter = factor_names.copy()
            st.rerun()
        
        if st.button("Clear All", key="clear_all_factors"):
            st.session_state.factor_filter = []
            st.rerun()
        
        st.divider()
        
        # Display options
        st.subheader("🎨 Display")
        annualized = st.toggle("Annualized", value=True, key="annualized_toggle")
        show_percentage = st.toggle("Show % of total", value=True, key="percentage_toggle") 
        
        currency = data_loader.data.get('time_series', {}).get('currency', 'USD')
        st.text(f"Currency: {currency}")
        
        st.divider()
        
        # Data refresh controls
        st.divider()
        render_data_refresh_controls(data_loader)
        
        st.divider()
        
        # Downloads section
        st.subheader("💾 Downloads")
        if st.button("Current View JSON", key="download_current"):
            st.info("Download functionality coming soon")
        if st.button("Full Payload", key="download_full"):
            st.info("Download functionality coming soon")
        
        # Validation status
        validation = data_loader.get_validation_info()
        checks = validation.get('checks', {})
        passes = checks.get('passes', False)
        
        st.divider()
        st.subheader("✅ Status")
        if passes:
            st.success("All checks passed")
        else:
            st.warning("Some checks failed")
        
        # Hierarchical data summary (if available)
        hierarchical_summary = data_loader.get_hierarchical_data_summary()
        if hierarchical_summary.get('total_components', 0) > 0:
            st.divider()
            st.subheader("🏗️ Data Summary")
            st.text(f"Components: {hierarchical_summary['total_components']}")
            st.text(f"With matrices: {hierarchical_summary['components_with_matrices']}")
            st.text(f"Schema: v{hierarchical_summary['schema_version']}")
            
            # Show which components have how many lenses
            lens_counts = hierarchical_summary.get('component_lens_counts', {})
            if lens_counts:
                max_lenses = max(lens_counts.values()) if lens_counts else 0
                components_full = sum(1 for count in lens_counts.values() if count >= 3)
                st.text(f"Full analysis: {components_full}/{len(lens_counts)}")
    
    return SidebarState(
        lens=lens,
        selected_node=selected_node,
        date_range=date_range,
        selected_factors=selected_factors,
        annualized=annualized,
        show_percentage=show_percentage
    )