import streamlit as st

def render_factor_lens_tab(data_loader, sidebar_state):
    """Render Tab 5 - Factor lens using direct schema data access"""
    
    st.header("Factor Lens - Factor Analysis")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # Get schema data for current component and lens
    schema_data = data_loader.get_schema_data(sidebar_state.selected_node, sidebar_state.lens)
    
    if not schema_data:
        st.info(f"Factor data not available for {sidebar_state.selected_node} - {sidebar_state.lens} lens")
        return
    
    # Factor selection info
    render_factor_selection_info(sidebar_state, data_loader)
    
    st.divider()
    
    # Factor contributions
    render_factor_contributions_direct(schema_data, sidebar_state)
    
    st.divider()
    
    # Factor exposures
    render_factor_exposures_direct(schema_data, sidebar_state)
    
    st.divider()
    
    # Additional factor data
    render_additional_factor_data_direct(schema_data, sidebar_state)

def render_factor_selection_info(sidebar_state, data_loader):
    """Display current factor selection info using direct schema data access"""
    
    st.subheader("Factor Selection")
    
    all_factors = data_loader.get_factor_names()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if sidebar_state.selected_factors:
            st.metric("Selected Factors", len(sidebar_state.selected_factors))
            st.markdown("**Active Factors:**")
            for factor in sidebar_state.selected_factors:
                st.markdown(f"• {factor}")
        else:
            st.metric("Selected Factors", "All")
            st.markdown("**All factors included in analysis**")
    
    with col2:
        st.metric("Total Available", len(all_factors))

def render_factor_contributions_direct(schema_data, sidebar_state):
    """Render factor contributions using direct schema data."""
    
    st.subheader("Factor Contributions")
    
    factor_contributions = schema_data.get('factor_contributions', {})
    
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
            
            for factor, contribution in sorted_contribs:
                st.markdown(f"**{factor}**: {contribution}")
        else:
            st.info("No factor contribution data for selected filters")
    else:
        st.info("Factor contributions not available")

def render_factor_exposures_direct(schema_data, sidebar_state):
    """Render factor exposures using direct schema data."""
    
    st.subheader("Factor Exposures")
    
    factor_exposures = schema_data.get('factor_exposures', {})
    
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
            
            for factor, exposure in sorted_exposures:
                st.markdown(f"**{factor}**: {exposure}")
        else:
            st.info("No factor exposure data for selected filters")
    else:
        st.info("Factor exposures not available")

def render_additional_factor_data_direct(schema_data, sidebar_state):
    """Render additional factor-related data using direct schema data."""
    
    st.subheader("Additional Factor Data")
    
    # Show weighted betas if available
    weighted_betas = schema_data.get('weighted_betas', {})
    if weighted_betas:
        st.markdown("**Weighted Betas**")
        
        # Filter by selected factors if any
        if sidebar_state.selected_factors:
            weighted_betas = {
                f: v for f, v in weighted_betas.items() 
                if f in sidebar_state.selected_factors
            }
        
        if weighted_betas:
            # Sort by absolute value
            sorted_betas = sorted(
                weighted_betas.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            for factor, beta in sorted_betas:
                st.markdown(f"**{factor}**: {beta}")
        else:
            st.info("No weighted beta data for selected filters")
    
    # Show decomposer results if available
    decomposer_results = schema_data.get('decomposer_results', {})
    if decomposer_results:
        st.markdown("**Decomposer Results**")
        
        # Show risk metrics related to factors
        factor_related_metrics = [
            'factor_risk_contribution',
            'factor_risk_percentage',
            'factor_variance',
            'factor_covariance'
        ]
        
        available_factor_metrics = {}
        for metric in factor_related_metrics:
            value = decomposer_results.get(metric)
            if value is not None:
                available_factor_metrics[metric] = value
        
        if available_factor_metrics:
            for metric, value in available_factor_metrics.items():
                st.markdown(f"**{metric.replace('_', ' ').title()}**: {value}")
        else:
            st.info("No factor-specific decomposer metrics available")
    
    # Show any other factor-related arrays or matrices
    factor_related_keys = [
        'factor_covariance_matrix',
        'factor_loadings',
        'factor_scores',
        'factor_risk_contributions'
    ]
    
    available_factor_data = {}
    for key in factor_related_keys:
        value = schema_data.get(key)
        if value is not None:
            available_factor_data[key] = value
    
    if available_factor_data:
        st.markdown("**Other Factor Data**")
        for key, value in available_factor_data.items():
            if isinstance(value, dict):
                st.markdown(f"**{key.replace('_', ' ').title()}**: {len(value)} entries")
            elif hasattr(value, '__len__'):
                st.markdown(f"**{key.replace('_', ' ').title()}**: {len(value)} elements")
            else:
                st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
    else:
        st.info("No additional factor data available")



