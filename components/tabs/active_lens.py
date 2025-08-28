import streamlit as st
import plotly.graph_objects as go

def render_active_lens_tab(data_loader, sidebar_state):
    """Render Tab 2 - Active lens using direct schema data access"""
    
    st.header("Active Lens - Portfolio vs Benchmark Differences")
    st.markdown(f"**Current Node:** {sidebar_state.selected_node}")
    
    # Get active schema data
    active_data = data_loader.get_schema_data(sidebar_state.selected_node, "active")
    
    if not active_data:
        st.info(f"Active risk data not available for {sidebar_state.selected_node}")
        return
    
    # Active KPIs
    render_active_kpis(active_data)
    
    st.divider()
    
    # Factor analysis
    render_active_factor_analysis(active_data, sidebar_state)
    
    st.divider()
    
    # Weights and exposures
    render_active_weights_exposures(active_data, sidebar_state)

def render_active_kpis(active_data):
    """Render active KPIs using direct schema data."""
    st.subheader("Active Risk KPIs")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_risk = active_data.get('total_risk', 0)
        st.metric("Active Risk", f"{total_risk}", help="Total active risk vs benchmark")
    
    with col2:
        factor_contrib = active_data.get('factor_risk_contribution', 0)
        st.metric("Factor Risk", f"{factor_contrib}", help="Active factor risk contribution")
    
    with col3:
        specific_contrib = active_data.get('specific_risk_contribution', 0)
        st.metric("Specific Risk", f"{specific_contrib}", help="Active specific risk contribution")
    
    with col4:
        # Show factor percentage if available
        factor_pct = active_data.get('factor_risk_percentage', 'N/A')
        st.metric("Factor %", f"{factor_pct}", help="Factor risk as % of total active risk")

def render_active_factor_analysis(active_data, sidebar_state):
    """Render active factor contributions and exposures"""
    
    st.subheader("Active Factor Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Factor Contributions**")
        factor_contributions = active_data.get('factor_contributions', {})
        
        if factor_contributions:
            # Filter by selected factors if any
            if sidebar_state.selected_factors:
                factor_contributions = {
                    f: v for f, v in factor_contributions.items() 
                    if f in sidebar_state.selected_factors
                }
            
            if factor_contributions:
                # Sort by absolute value
                sorted_contribs = sorted(
                    factor_contributions.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                for factor, contribution in sorted_contribs[:10]:
                    st.markdown(f"**{factor}**: {contribution}")
            else:
                st.info("No factor contribution data for selected filters")
        else:
            st.info("Factor contributions not available")
    
    with col2:
        st.markdown("**Factor Exposures**")
        factor_exposures = active_data.get('factor_exposures', {})
        
        if factor_exposures:
            # Filter by selected factors if any
            if sidebar_state.selected_factors:
                factor_exposures = {
                    f: v for f, v in factor_exposures.items() 
                    if f in sidebar_state.selected_factors
                }
            
            if factor_exposures:
                # Sort by absolute value
                sorted_exposures = sorted(
                    factor_exposures.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                for factor, exposure in sorted_exposures[:10]:
                    st.markdown(f"**{factor}**: {exposure}")
            else:
                st.info("No factor exposure data for selected filters")
        else:
            st.info("Factor exposures not available")

def render_active_weights_exposures(active_data, sidebar_state):
    """Render active weights and additional data"""
    
    st.subheader("Active Weights & Additional Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Active Weights**")
        active_weights = active_data.get('active_weights', {})
        
        if active_weights:
            # Sort by absolute weight
            sorted_weights = sorted(
                active_weights.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:10]
            
            for asset, weight in sorted_weights:
                st.markdown(f"**{asset}**: {weight}")
        else:
            st.info("Active weights not available")
    
    with col2:
        st.markdown("**Portfolio Weights**")
        portfolio_weights = active_data.get('portfolio_weights', {})
        
        if portfolio_weights:
            # Sort by weight
            sorted_weights = sorted(
                portfolio_weights.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:10]
            
            for asset, weight in sorted_weights:
                st.markdown(f"**{asset}**: {weight}")
        else:
            st.info("Portfolio weights not available")
    
    # Show additional metrics if available
    st.subheader("Additional Active Risk Metrics")
    
    metrics_to_show = [
        'total_active_risk', 'allocation_factor_risk', 'selection_factor_risk',
        'total_allocation_risk', 'total_selection_risk'
    ]
    
    available_metrics = {}
    for metric in metrics_to_show:
        value = active_data.get(metric)
        if value is not None:
            available_metrics[metric] = value
    
    if available_metrics:
        cols = st.columns(len(available_metrics))
        for i, (metric, value) in enumerate(available_metrics.items()):
            with cols[i]:
                st.metric(metric.replace('_', ' ').title(), f"{value}")
    else:
        st.info("No additional active risk metrics available")
    
    # Show decomposer results if available
    decomposer_results = active_data.get('decomposer_results', {})
    if decomposer_results:
        st.subheader("Decomposer Results")
        for key, value in decomposer_results.items():
            st.markdown(f"**{key}**: {value}")
    
    # Show weighted betas if available
    weighted_betas = active_data.get('weighted_betas', {})
    if weighted_betas:
        st.subheader("Weighted Betas")
        sorted_betas = sorted(
            weighted_betas.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:10]
        
        for factor, beta in sorted_betas:
            st.markdown(f"**{factor}**: {beta}")