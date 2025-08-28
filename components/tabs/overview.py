import pandas as pd
import streamlit as st
import plotly.express as px

def render_overview_tab(data_loader, sidebar_state):
    """Render Tab 1 - Overview (snapshot) using direct schema data access"""
    
    st.header("Overview")
    st.markdown(f"**Node:** {sidebar_state.selected_node}")
    
    # Compound returns time series chart
    render_compound_returns_chart(data_loader, sidebar_state)
    
    st.divider()
    
    # Risk composition - stacked bars for Portfolio, Benchmark, Active
    render_risk_composition(data_loader, sidebar_state)
    
    

def render_compound_returns_chart(data_loader, sidebar_state):
    """Render compound returns time series chart using direct schema data."""
    st.subheader("Compound Returns")
    
    # Get schema data for current component and lens
    portfolio_data = data_loader.get_schema_data(sidebar_state.selected_node, "portfolio")
    benchmark_data = data_loader.get_schema_data(sidebar_state.selected_node, "benchmark")
    active_data = data_loader.get_schema_data(sidebar_state.selected_node, "active")
    
    # Extract time series data and dates from schema
    portfolio_returns = portfolio_data.get('portfolio_return', [])
    portfolio_dates = portfolio_data.get('portfolio_return_dates', [])
    
    benchmark_returns = benchmark_data.get('benchmark_return', [])
    benchmark_dates = benchmark_data.get('benchmark_return_dates', [])

    if portfolio_returns and portfolio_dates and len(portfolio_returns) == len(portfolio_dates):
        portfolio = pd.Series(index=portfolio_dates, data=portfolio_returns)
    else:
        portfolio = pd.Series()
    
    if benchmark_returns and benchmark_dates and len(benchmark_returns) == len(benchmark_dates):
        benchmark = pd.Series(index=benchmark_dates, data=benchmark_returns)
    else:
        benchmark = pd.Series()

    combined = pd.DataFrame({
        'Portfolio': portfolio,
        'Benchmark': benchmark,
    }).add(1).cumprod().add(-1)
    
    if not combined.empty:
        combined['Active'] = combined['Portfolio'] - combined['Benchmark']
        fig = px.line(combined, title="", labels={'value': 'Cumulative Return', 'index': 'Date', 'variable': ''})
        # axis format should be in percent with 1 decimal
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time series data available for compound returns chart")


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



