import pandas as pd
import streamlit as st
import plotly.express as px

def render_overview_tab(data_loader, sidebar_state):
    """Render Tab 1 - Overview (snapshot) using direct schema data access"""
    

    st.header(f"**Node:** {sidebar_state.selected_node}")
    
    # Compound returns time series chart
    render_compound_returns_chart(data_loader, sidebar_state)
    
    st.divider()
    
    # Risk composition - stacked bars for Portfolio, Benchmark, Active
    render_risk_composition(data_loader, sidebar_state)
    
    

def render_compound_returns_chart(data_loader, sidebar_state):
    """Render compound returns time series chart using cached time series data."""
    st.subheader("Compound Returns")
    
    # Get time series data for selected component
    time_series_df = data_loader.get_time_series_for_components([sidebar_state.selected_node])
    
    if time_series_df.empty:
        st.info("No time series data available for compound returns chart")
        return
    
    # Pivot long-form data to wide-form for plotting
    pivoted = time_series_df.pivot(index='date', columns='lens', values='return_value')
    
    if pivoted.empty:
        st.info("No time series data available for compound returns chart")
        return
    
    combined = pivoted.add(1).cumprod().add(-1)
    combined["active"] = combined["portfolio"] - combined["benchmark"]

    # Rename columns to match expected format
    column_mapping = {
        'portfolio': 'Portfolio',
        'benchmark': 'Benchmark', 
        'active': 'Active'
    }
    combined = combined.rename(columns=column_mapping)
    
    # Plot the compound returns
    fig = px.line(combined, title="", labels={'value': 'Cumulative Return', 'index': 'Date', 'variable': ''})
    # axis format should be in percent with 1 decimal
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)


def render_risk_composition(data_loader, sidebar_state):
    """Render risk composition chart using direct schema data."""
    st.subheader("Risk Composition")
    
    # Get data for all lenses
    portfolio_data = data_loader.get_schema_data(sidebar_state.selected_node, "portfolio")
    benchmark_data = data_loader.get_schema_data(sidebar_state.selected_node, "benchmark")
    active_data = data_loader.get_schema_data(sidebar_state.selected_node, "active")
    
    if portfolio_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Portfolio**")
            total = portfolio_data.get('total_risk', 0)
            factor = portfolio_data.get('factor_risk_contribution', 0)
            specific = portfolio_data.get('specific_risk_contribution', 0)
            st.metric("Total", f"{total:.1%}")
            st.caption(f"Factor: {factor:.1%} | Specific: {specific:.1%}")
        
        with col2:
            if benchmark_data:
                st.markdown("**Benchmark**")
                total = benchmark_data.get('total_risk', 0)
                factor = benchmark_data.get('factor_risk_contribution', 0)
                specific = benchmark_data.get('specific_risk_contribution', 0)
                st.metric("Total", f"{total:.1%}")
                st.caption(f"Factor: {factor:.1%} | Specific: {specific:.1%}")
            else:
                st.info("Benchmark data not available")
        
        with col3:
            if active_data:
                st.markdown("**Active**")
                total = active_data.get('total_risk', 0)
                factor = active_data.get('factor_risk_contribution', 0)
                specific = active_data.get('specific_risk_contribution', 0)
                st.metric("Total", f"{total:.1%}")
                st.caption(f"Factor: {factor:.1%} | Specific: {specific:.1%}")
            else:
                st.info("Active data not available")



