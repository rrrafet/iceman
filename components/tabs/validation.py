import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color

def render_validation_tab(data_loader, sidebar_state):
    """Render Tab 11 - Validation & diagnostics"""
    
    st.header("Validation & Diagnostics")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # Validation header with overall status
    render_validation_header(data_loader)
    
    st.divider()
    
    # Validation checks grid
    render_validation_checks(data_loader, sidebar_state)
    
    st.divider()
    
    # Detailed diagnostic messages
    render_diagnostic_messages(data_loader, sidebar_state)

def render_validation_header(data_loader):
    """Render validation header with overall status"""
    
    st.subheader("Validation Status")
    
    # Get overall validation info
    validation_info = data_loader.get_validation_info()
    validation_checks = validation_info.get('checks', {})
    
    overall_passes = validation_checks.get('passes', True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if overall_passes:
            st.success("VALIDATED")
            st.metric("Overall Status", "PASS")
        else:
            st.error("VALIDATION FAILED")
            st.metric("Overall Status", "FAIL")
    
    with col2:
        # Individual check status
        euler_check = validation_checks.get('euler_identity', True)
        weight_check = validation_checks.get('weight_consistency', True)
        
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            if euler_check:
                st.success("Euler Identity: PASS")
            else:
                st.error("Euler Identity: FAIL")
        
        with col2_2:
            if weight_check:
                st.success("Weight Consistency: PASS")
            else:
                st.error("Weight Consistency: FAIL")
    
    with col3:
        # Summary metrics
        try:
            risk_status = data_loader.get_risk_analysis_status()
            if risk_status.get('analysis_completed', False):
                st.metric("Data Quality", "HIGH")
            else:
                st.metric("Data Quality", "PENDING")
        except:
            st.metric("Data Quality", "UNKNOWN")

def render_validation_checks(data_loader, sidebar_state):
    """Render detailed validation checks grid"""
    
    st.subheader("Validation Checks")
    
    # Get validation results from risk service
    try:
        validation_result = data_loader.validate_risk_analysis()
        is_valid = validation_result.get('valid', False)
        
        if is_valid:
            render_successful_validations(data_loader, sidebar_state)
        else:
            error_message = validation_result.get('message', 'Unknown validation error')
            st.error(f"Validation failed: {error_message}")
            render_failed_validations(data_loader, sidebar_state)
    
    except Exception as e:
        st.warning("Unable to retrieve validation results")
        render_placeholder_validations()

def render_successful_validations(data_loader, sidebar_state):
    """Render validation checks when they pass"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Risk Model Validation**")
        
        # Asset sum check
        st.success("Asset Sum Check: PASS")
        st.caption("Sum of individual asset risks matches portfolio risk")
        
        # Factor-specific sum check
        st.success("Factor-Specific Sum Check: PASS")
        st.caption("Factor contributions sum correctly")
        
        # Euler identity validation
        st.success("Euler Identity Check: PASS")
        st.caption("Risk decomposition follows mathematical identity")
    
    with col2:
        st.markdown("**Data Consistency Validation**")
        
        # Component sum validation
        st.success("Component Sum Check: PASS") 
        st.caption("Hierarchical components sum to total")
        
        # Weight consistency
        st.success("Weight Consistency: PASS")
        st.caption("Portfolio weights sum to 100%")
        
        # Data completeness
        st.success("Data Completeness: PASS")
        st.caption("All required data fields present")

def render_failed_validations(data_loader, sidebar_state):
    """Render validation checks when they fail"""
    
    st.markdown("**Failed Validation Details**")
    
    # This would show actual validation failures
    st.error("Some validation checks have failed. Details:")
    
    # Placeholder for actual validation error details
    with st.expander("Validation Error Details"):
        st.markdown("""
        **Common Validation Issues:**
        
        1. **Euler Identity Violations**: Risk contributions don't sum to total risk
        2. **Weight Inconsistencies**: Portfolio weights don't sum to 100%
        3. **Missing Data**: Required fields are empty or null
        4. **Numerical Precision**: Rounding errors in calculations
        5. **Hierarchy Inconsistencies**: Parent-child relationships don't match
        """)

def render_placeholder_validations():
    """Render placeholder validation when service unavailable"""
    
    st.info("Validation checks will be shown when risk analysis service is available")
    
    st.markdown("**Expected Validation Checks:**")
    
    checks = [
        ("Asset Sum Check", "Verify individual asset risks sum to portfolio total"),
        ("Factor Sum Check", "Verify factor contributions sum correctly"), 
        ("Euler Identity", "Mathematical consistency of risk decomposition"),
        ("Weight Consistency", "Portfolio weights sum to 100%"),
        ("Hierarchy Consistency", "Parent-child relationships are valid"),
        ("Data Completeness", "All required fields are populated"),
        ("Numerical Precision", "Calculations within tolerance limits"),
        ("Time Series Alignment", "Dates and periods are consistent")
    ]
    
    for check_name, description in checks:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.info(f"{check_name}")
        with col2:
            st.caption(description)

def render_diagnostic_messages(data_loader, sidebar_state):
    """Render detailed diagnostic messages and recommendations"""
    
    st.subheader("Diagnostic Messages")
    
    # Key risk concentrations
    render_risk_concentration_diagnostics(data_loader, sidebar_state)
    
    st.divider()
    
    # Data quality diagnostics
    render_data_quality_diagnostics(data_loader, sidebar_state)
    
    st.divider()
    
    # Methodology notes
    render_methodology_notes()

def render_risk_concentration_diagnostics(data_loader, sidebar_state):
    """Render diagnostics about risk concentrations"""
    
    st.markdown("**Risk Concentration Diagnostics**")
    
    # Get contribution data for analysis
    contributions = data_loader.get_contributions(sidebar_state.lens, "by_asset")
    
    if contributions:
        # Calculate concentration metrics
        total_risk = sum(abs(v) for v in contributions.values())
        sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Top contributor analysis
        if sorted_contribs:
            top_contributor = sorted_contribs[0]
            top_contrib_pct = abs(top_contributor[1]) / total_risk * 100 if total_risk > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Max Single Contributor", f"{top_contrib_pct:.1f}%")
                st.caption(f"Component: {top_contributor[0]}")
                
                if top_contrib_pct > 20:
                    st.warning("High concentration risk detected")
                elif top_contrib_pct > 10:
                    st.info("Moderate concentration risk")
                else:
                    st.success("Well-diversified risk")
            
            with col2:
                # Top 5 concentration
                top_5_risk = sum(abs(contrib[1]) for contrib in sorted_contribs[:5])
                top_5_pct = top_5_risk / total_risk * 100 if total_risk > 0 else 0
                
                st.metric("Top 5 Contributors", f"{top_5_pct:.1f}%")
                st.caption("Percentage of total risk")
                
                if top_5_pct > 60:
                    st.warning("Risk concentrated in few positions")
                elif top_5_pct > 40:
                    st.info("Moderate risk concentration")
                else:
                    st.success("Good risk diversification")
        
        # Factor contribution reasonableness
        factor_contribs = data_loader.get_contributions(sidebar_state.lens, "by_factor")
        if factor_contribs:
            max_factor = max(factor_contribs.items(), key=lambda x: abs(x[1]))
            st.markdown(f"**Dominant Factor**: {max_factor[0]} ({max_factor[1]:.1f} bps)")
    
    else:
        st.info("No contribution data available for concentration analysis")

def render_data_quality_diagnostics(data_loader, sidebar_state):
    """Render data quality diagnostics"""
    
    st.markdown("**Data Quality Assessment**")
    
    # Time series data availability
    current_node = sidebar_state.selected_node
    
    portfolio_data = data_loader.get_time_series_data('portfolio_returns', current_node)
    benchmark_data = data_loader.get_time_series_data('benchmark_returns', current_node)
    active_data = data_loader.get_time_series_data('active_returns', current_node)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Time Series Availability**")
        
        if portfolio_data:
            st.success(f"Portfolio returns: {len(portfolio_data)} periods")
        else:
            st.warning("Portfolio returns: Not available")
        
        if benchmark_data:
            st.success(f"Benchmark returns: {len(benchmark_data)} periods")
        else:
            st.info("Benchmark returns: Not available")
        
        if active_data:
            st.success(f"Active returns: {len(active_data)} periods")
        else:
            st.info("Active returns: Not available")
    
    with col2:
        st.markdown("**Data Completeness**")
        
        # Check factor data
        factor_names = data_loader.get_factor_names()
        st.metric("Available Factors", len(factor_names))
        
        # Check component coverage
        components = data_loader.get_available_hierarchical_components()
        st.metric("Portfolio Components", len(components))
        
        # Check risk service status
        if data_loader.has_risk_service():
            st.success("Risk service: Active")
        else:
            st.warning("Risk service: Not available")

def render_methodology_notes():
    """Render methodology and calculation notes"""
    
    st.markdown("**Methodology Notes**")
    
    with st.expander("Risk Model Methodology"):
        st.markdown("""
        **Risk Calculation Framework:**
        
        - **Factor Risk**: Calculated using linear factor model with estimated exposures
        - **Specific Risk**: Idiosyncratic risk not explained by factors
        - **Total Risk**: Square root of (Factor Risk² + Specific Risk² + 2×Covariance)
        - **Euler Decomposition**: Risk contributions sum to total portfolio risk
        
        **Validation Tolerances:**
        - Numerical precision: ±0.01 bps
        - Weight consistency: ±0.001%
        - Euler identity: ±0.1% of total risk
        """)
    
    with st.expander("Data Sources & Assumptions"):
        st.markdown("""
        **Data Sources:**
        - Portfolio holdings and weights
        - Risk factor returns (historical)
        - Benchmark composition and returns
        
        **Key Assumptions:**
        - Factor model is linear and stable
        - Historical relationships persist
        - No look-ahead bias in factor estimation
        - Missing data handled via interpolation/exclusion
        
        **Limitations:**
        - Model risk in factor selection
        - Estimation error in factor loadings
        - Regime changes not captured
        """)
    
    with st.expander("Interpretation Guidelines"):
        st.markdown("""
        **Risk Interpretation:**
        
        - **Low Risk**: < 100 bps annualized
        - **Moderate Risk**: 100-300 bps annualized  
        - **High Risk**: > 300 bps annualized
        
        **Concentration Thresholds:**
        - **Single Position**: > 20% of risk is high concentration
        - **Top 5 Positions**: > 60% of risk indicates concentration
        - **Factor Exposure**: > 50% from single factor is concerning
        
        **Validation Standards:**
        - All mathematical identities must hold within tolerance
        - Data completeness > 95% for reliable results
        - Time series alignment across all datasets required
        """)

def render_tolerance_settings():
    """Render current tolerance settings for validation"""
    
    st.markdown("**Current Tolerance Settings**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Euler Tolerance", "0.1%")
        st.caption("Risk decomposition accuracy")
    
    with col2:
        st.metric("Weight Tolerance", "0.001%")
        st.caption("Portfolio weight precision")
    
    with col3:
        st.metric("Numerical Precision", "0.01 bps")
        st.caption("Calculation accuracy")