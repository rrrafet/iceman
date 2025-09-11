import pandas as pd
import streamlit as st
import plotly.express as px

from spark.ui.apps.maverick.utils.colors import get_chart_color

def render_overview_tab(data_access_service, sidebar_state):
    """Render Tab 1 - Overview (snapshot) using 3-layer architecture services"""
    
    st.header(f"**Component:** {sidebar_state.selected_component_id}")
    
    # Risk composition - stacked bars for Portfolio, Benchmark, Active
    render_risk_composition(data_access_service, sidebar_state)

    st.divider()
    # Compound returns time series chart
    render_compound_returns_chart(data_access_service, sidebar_state)
    
    

    
    

def render_compound_returns_chart(data_access_service, sidebar_state):
    """Render compound returns time series chart using cached time series data."""
    st.subheader("Compound Returns")
    
    # Get time series data for selected component
    try:
        portfolio_returns = data_access_service.get_portfolio_returns(sidebar_state.selected_component_id)
        benchmark_returns = data_access_service.get_benchmark_returns(sidebar_state.selected_component_id)
        active_returns = data_access_service.get_active_returns(sidebar_state.selected_component_id)
        
        # Convert to cumulative returns
        cumulative_portfolio = (1 + portfolio_returns).cumprod() - 1
        cumulative_benchmark = (1 + benchmark_returns).cumprod() - 1
        cumulative_active = (1 + active_returns).cumprod() - 1
        
        # Combine into DataFrame
        combined = pd.DataFrame({
            'Portfolio': cumulative_portfolio,
            'Benchmark': cumulative_benchmark,
            'Active': cumulative_active
        })
    except Exception as e:
        st.error(f"Error loading time series data: {e}")
        return
    
    if combined.empty:
        st.info("No time series data available for compound returns chart")
        return
    
    # Plot the compound returns with proper colors
    fig = px.line(combined, title="", labels={'value': 'Cumulative Return', 'index': 'Date', 'variable': ''})
    
    # Apply consistent colors for Portfolio/Benchmark/Active
    color_map = {
        'Portfolio': get_chart_color('portfolio'),
        'Benchmark': get_chart_color('benchmark'), 
        'Active': get_chart_color('active')
    }
    
    # Update trace colors
    for trace in fig.data:
        if trace.name in color_map:
            trace.line.color = color_map[trace.name]
    
    # axis format should be in percent with 1 decimal
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width='stretch')


def render_risk_composition(data_access_service, sidebar_state):
    """Render risk composition chart using direct schema data."""
    st.subheader("Risk Composition")
    
    # Get risk data for all lenses
    component_id = sidebar_state.selected_component_id
    
    try:
        portfolio_risk = data_access_service.get_risk_decomposition(component_id, "portfolio")
        benchmark_risk = data_access_service.get_risk_decomposition(component_id, "benchmark")
        active_risk = data_access_service.get_risk_decomposition(component_id, "active")
    except Exception as e:
        st.error(f"Error loading risk data: {e}")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Portfolio**")
        total = portfolio_risk.get('total_risk', 0)
        factor = portfolio_risk.get('factor_risk_contribution', 0)
        specific = portfolio_risk.get('specific_risk_contribution', 0)
        st.metric("Total", f"{total:.1%}")
        st.caption(f"Factor: {factor:.1%} | Specific: {specific:.1%}")
    
    with col2:
        st.markdown("**Benchmark**")
        total = benchmark_risk.get('total_risk', 0)
        factor = benchmark_risk.get('factor_risk_contribution', 0)
        specific = benchmark_risk.get('specific_risk_contribution', 0)
        st.metric("Total", f"{total:.1%}")
        st.caption(f"Factor: {factor:.1%} | Specific: {specific:.1%}")
    
    with col3:
        st.markdown("**Active**")
        total = active_risk.get('total_risk', 0)
        factor = active_risk.get('factor_risk_contribution', 0)
        specific = active_risk.get('specific_risk_contribution', 0)
        st.metric("Total", f"{total:.1%}")
        st.caption(f"Factor: {factor:.1%} | Specific: {specific:.1%}")



