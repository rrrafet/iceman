import streamlit as st
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.colors import get_chart_color

def render_decomposition_tab(data_loader, sidebar_state):
    """Render Tab 8 - Decomposition (allocation / selection)"""
    
    st.header("Decomposition - Allocation & Selection Effects")
    st.markdown(f"**Current View:** {sidebar_state.lens.title()} | **Node:** {sidebar_state.selected_node}")
    
    # Check for decomposition data availability
    render_decomposition_status(data_loader)
    
    st.divider()
    
    # Placeholder design for when data is available
    render_decomposition_design(data_loader, sidebar_state)

def render_decomposition_status(data_loader):
    """Show status of decomposition data availability using hierarchical schema"""
    
    st.subheader("Decomposition Status")
    
    # NEW: Check for decomposition data using hierarchical schema methods
    try:
        # Get comprehensive schema data
        schema_data = data_loader.get_comprehensive_schema_data("TOTAL")
        
        if schema_data:
            # Check hierarchical validation status
            validation = data_loader.get_hierarchical_validation()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Data Availability**")
                components_with_data = validation.get('components_with_data', 0)
                
                if components_with_data > 0:
                    st.success(f"Hierarchical Data: Available")
                    st.metric("Components with Data", components_with_data)
                else:
                    st.info("Hierarchical Data: Not Available")
            
            with col2:
                st.markdown("**Active Analysis**")
                active_decomposition = data_loader.get_component_decomposition("TOTAL", "active")
                
                if active_decomposition:
                    st.success("Active Decomposition: Available")
                    active_factors = active_decomposition.get('factor_contributions', {})
                    st.metric("Active Factors", len(active_factors))
                else:
                    st.info("Active Decomposition: Not Available")
            
            with col3:
                st.markdown("**Portfolio vs Benchmark**")
                portfolio_decomp = data_loader.get_component_decomposition("TOTAL", "portfolio")
                benchmark_decomp = data_loader.get_component_decomposition("TOTAL", "benchmark")
                
                if portfolio_decomp and benchmark_decomp:
                    st.success("Attribution Ready: Available")
                    st.metric("Lenses Available", "2/3")
                elif portfolio_decomp or benchmark_decomp:
                    st.warning("Attribution Ready: Partial")
                    st.metric("Lenses Available", "1/3")
                else:
                    st.info("Attribution Ready: Not Available")
        else:
            st.warning("Schema data not available - check risk analysis")
            
        # Show component-level data availability
        all_components = data_loader.get_all_component_risk_results()
        if all_components:
            st.markdown("**Component-Level Data:**")
            available_components = len(all_components)
            st.info(f"Risk data available for {available_components} components")
            
            # Show lens coverage
            lens_coverage = validation.get('lens_coverage', {})
            if lens_coverage:
                coverage_text = ", ".join([f"{lens}: {count}" for lens, count in lens_coverage.items()])
                st.caption(f"Lens coverage: {coverage_text}")
                
    except Exception as e:
        st.error(f"Unable to check decomposition status: {str(e)}")
        st.info("This may indicate that risk analysis has not been run yet.")
    
    # Enhanced explanation with hierarchical context
    st.markdown("""
    **Hierarchical Decomposition Analysis:**
    
    Our framework supports multi-level performance attribution:
    - **Factor-Based Attribution**: Risk factor contributions to active returns
    - **Component-Level Attribution**: Hierarchical breakdown across portfolio levels  
    - **Asset-Level Attribution**: Individual security contribution analysis
    - **Cross-Component Effects**: Interaction effects between hierarchy levels
    
    The hierarchical schema enables attribution analysis at any portfolio component level,
    providing insights into both factor tilts and component selection decisions.
    """)

def render_decomposition_design(data_loader, sidebar_state):
    """Render placeholder design for decomposition analysis"""
    
    st.subheader("Decomposition Analysis")
    
    # Create tabs for different views
    decomp_tabs = st.tabs(["Summary", "By Factor", "By Component", "Waterfall"])
    
    with decomp_tabs[0]:
        render_decomposition_summary()
    
    with decomp_tabs[1]:
        render_decomposition_by_factor(data_loader, sidebar_state)
    
    with decomp_tabs[2]:
        render_decomposition_by_component(data_loader, sidebar_state)
    
    with decomp_tabs[3]:
        render_decomposition_waterfall(data_loader, sidebar_state)

def render_decomposition_summary():
    """Render summary view of decomposition effects"""
    
    st.markdown("**Effect Summary**")
    
    # Placeholder data structure
    st.info("""
    **When decomposition data is available, this section will show:**
    
    - Total active return broken down by effect
    - Statistical significance of each effect
    - Contribution to tracking error from each source
    - Time series evolution of effects
    """)
    
    # Mock structure for visualization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Allocation Effect**")
        st.metric("Total Impact", "--- bps", help="Coming soon")
        st.markdown("Top contributors will be listed here")
    
    with col2:
        st.markdown("**Selection Effect**") 
        st.metric("Total Impact", "--- bps", help="Coming soon")
        st.markdown("Top contributors will be listed here")
    
    with col3:
        st.markdown("**Interaction Effect**")
        st.metric("Total Impact", "--- bps", help="Coming soon")
        st.markdown("Combined effects will be shown here")

def render_decomposition_by_factor(data_loader, sidebar_state):
    """Render decomposition analysis by factor using hierarchical data"""
    
    st.markdown("**Decomposition by Factor**")
    
    # NEW: Get factor contributions from hierarchical data
    portfolio_factors = data_loader.get_factor_contributions_from_schema(sidebar_state.selected_node, "portfolio")
    benchmark_factors = data_loader.get_factor_contributions_from_schema(sidebar_state.selected_node, "benchmark")
    active_factors = data_loader.get_factor_contributions_from_schema(sidebar_state.selected_node, "active")
    
    if not portfolio_factors and not benchmark_factors and not active_factors:
        st.info("""
        **Factor-based Decomposition Analysis:**
        
        Factor attribution data not available for current component.
        
        When available, this view shows:
        - Factor contributions from portfolio vs benchmark exposures
        - Active factor tilts and their risk impact
        - Factor-specific allocation effects
        """)
        return
    
    # Create factor decomposition table with actual data
    import pandas as pd
    
    # Get all unique factors
    all_factors = set()
    if portfolio_factors:
        all_factors.update(portfolio_factors.keys())
    if benchmark_factors:
        all_factors.update(benchmark_factors.keys())
    if active_factors:
        all_factors.update(active_factors.keys())
    
    if all_factors:
        st.markdown("**Factor Attribution Table:**")
        
        table_data = []
        for factor in sorted(all_factors):
            portfolio_contrib = (portfolio_factors.get(factor, 0) * 10000) if portfolio_factors else 0
            benchmark_contrib = (benchmark_factors.get(factor, 0) * 10000) if benchmark_factors else 0
            active_contrib = (active_factors.get(factor, 0) * 10000) if active_factors else 0
            
            # Calculate implied allocation effect (difference)
            allocation_effect = portfolio_contrib - benchmark_contrib if benchmark_factors else 0
            
            table_data.append({
                'Factor': factor,
                'Portfolio (bps)': f"{portfolio_contrib:.1f}" if portfolio_factors else "N/A",
                'Benchmark (bps)': f"{benchmark_contrib:.1f}" if benchmark_factors else "N/A",
                'Active (bps)': f"{active_contrib:.1f}" if active_factors else "N/A",
                'Allocation Effect (bps)': f"{allocation_effect:.1f}" if benchmark_factors else "N/A"
            })
        
        factor_df = pd.DataFrame(table_data)
        st.dataframe(factor_df, use_container_width=True)
        
        # Add explanation of what we're showing
        st.info("""
        **Table Explanation:**
        - **Portfolio**: Factor risk contribution in the portfolio
        - **Benchmark**: Factor risk contribution in the benchmark  
        - **Active**: Direct active factor risk contribution
        - **Allocation Effect**: Implied allocation effect (Portfolio - Benchmark)
        """)
    else:
        st.info("No factor data available for decomposition analysis")

def render_decomposition_by_component(data_loader, sidebar_state):
    """Render decomposition analysis by component using hierarchical data"""
    
    st.markdown("**Decomposition by Component**")
    
    # NEW: Get child components and their risk data
    children = data_loader.get_component_children_from_schema(sidebar_state.selected_node)
    
    if not children:
        st.info("""
        **Component-based Decomposition Analysis:**
        
        No child components found for current node.
        
        When available, this view shows:
        - Component-level risk contributions
        - Portfolio vs benchmark weights by component
        - Component-specific active risk breakdown
        """)
        return
    
    st.markdown(f"**Analysis for children of {sidebar_state.selected_node}:**")
    
    # Get comprehensive schema data for weights
    schema_data = data_loader.get_comprehensive_schema_data(sidebar_state.selected_node)
    weights_section = schema_data.get('weights', {}) if schema_data else {}
    
    # Build component analysis table
    import pandas as pd
    table_data = []
    
    for component_id in children:
        # Get component risk summaries
        portfolio_decomp = data_loader.get_component_decomposition(component_id, "portfolio")
        benchmark_decomp = data_loader.get_component_decomposition(component_id, "benchmark")
        active_decomp = data_loader.get_component_decomposition(component_id, "active")
        
        # Calculate metrics
        portfolio_risk = (portfolio_decomp.get('total_risk', 0) * 10000) if portfolio_decomp else 0
        benchmark_risk = (benchmark_decomp.get('total_risk', 0) * 10000) if benchmark_decomp else 0
        active_risk = (active_decomp.get('total_risk', 0) * 10000) if active_decomp else 0
        
        # Get weights (if available)
        portfolio_weights = weights_section.get('portfolio_weights', {})
        benchmark_weights = weights_section.get('benchmark_weights', {})
        
        portfolio_weight = portfolio_weights.get(component_id, 0) * 100  # Convert to percentage
        benchmark_weight = benchmark_weights.get(component_id, 0) * 100
        
        # Calculate allocation effect (simplified)
        allocation_effect = (portfolio_weight - benchmark_weight) if benchmark_weight > 0 else 0
        
        table_data.append({
            'Component': component_id,
            'Portfolio Weight (%)': f"{portfolio_weight:.2f}" if portfolio_weight > 0 else "N/A",
            'Benchmark Weight (%)': f"{benchmark_weight:.2f}" if benchmark_weight > 0 else "N/A", 
            'Portfolio Risk (bps)': f"{portfolio_risk:.0f}" if portfolio_decomp else "N/A",
            'Benchmark Risk (bps)': f"{benchmark_risk:.0f}" if benchmark_decomp else "N/A",
            'Active Risk (bps)': f"{active_risk:.0f}" if active_decomp else "N/A",
            'Weight Tilt (%)': f"{allocation_effect:.2f}" if benchmark_weight > 0 else "N/A"
        })
    
    if table_data:
        component_df = pd.DataFrame(table_data)
        st.dataframe(component_df, use_container_width=True)
        
        st.info("""
        **Table Explanation:**
        - **Portfolio/Benchmark Risk**: Total risk contribution from each component
        - **Active Risk**: Active risk contribution (portfolio vs benchmark)
        - **Weight Tilt**: Over/under-weighting relative to benchmark
        """)
        
        # Show component with highest active risk
        if any(row['Active Risk (bps)'] != 'N/A' for row in table_data):
            valid_active_risks = [(row['Component'], float(row['Active Risk (bps)'])) 
                                 for row in table_data if row['Active Risk (bps)'] != 'N/A']
            if valid_active_risks:
                top_component = max(valid_active_risks, key=lambda x: abs(x[1]))
                st.success(f"Highest Active Risk: {top_component[0]} ({top_component[1]:.0f} bps)")
    else:
        st.info("No component data available for decomposition analysis")

def render_decomposition_waterfall(data_loader, sidebar_state):
    """Render waterfall chart for risk decomposition effects using hierarchical data"""
    
    st.markdown("**Risk Decomposition Waterfall**")
    
    # NEW: Get actual decomposition data
    portfolio_decomp = data_loader.get_component_decomposition(sidebar_state.selected_node, "portfolio")
    benchmark_decomp = data_loader.get_component_decomposition(sidebar_state.selected_node, "benchmark")
    active_decomp = data_loader.get_component_decomposition(sidebar_state.selected_node, "active")
    
    fig = go.Figure()
    
    if portfolio_decomp or benchmark_decomp or active_decomp:
        # Use actual data for waterfall
        portfolio_risk = (portfolio_decomp.get('total_risk', 0) * 10000) if portfolio_decomp else 0
        benchmark_risk = (benchmark_decomp.get('total_risk', 0) * 10000) if benchmark_decomp else 0
        
        portfolio_factor = (portfolio_decomp.get('factor_risk_contribution', 0) * 10000) if portfolio_decomp else 0
        portfolio_specific = (portfolio_decomp.get('specific_risk_contribution', 0) * 10000) if portfolio_decomp else 0
        
        categories = ['Benchmark Risk', 'Factor Effect', 'Specific Effect', 'Total Portfolio Risk']
        values = [benchmark_risk, portfolio_factor - (benchmark_decomp.get('factor_risk_contribution', 0) * 10000 if benchmark_decomp else 0), 
                 portfolio_specific, portfolio_risk]
        colors = [get_chart_color("benchmark"), get_chart_color("positive"), get_chart_color("negative"), get_chart_color("portfolio")]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.0f} bps" for v in values],
            textposition='outside',
            opacity=0.8
        ))
        
        title = f"Risk Decomposition: {sidebar_state.selected_node} ({sidebar_state.lens.title()} Lens)"
        note_text = "Risk decomposition from hierarchical schema data"
        
    else:
        # Fallback to placeholder
        categories = ['Benchmark', 'Allocation', 'Selection', 'Interaction', 'Portfolio']
        values = [5, 2, -1, 0.5, 6.5]  # Example values
        colors = ['lightblue', 'green', 'red', 'orange', 'darkblue']
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=['Base', '+2 bps', '-1 bps', '+0.5 bps', 'Total'],
            textposition='outside',
            opacity=0.3  # Low opacity to show this is placeholder
        ))
        
        title = "Example Decomposition Waterfall (Placeholder)"
        note_text = "Placeholder visualization - run risk analysis to see actual data"
    
    fig.update_layout(
        title=title,
        xaxis_title="Risk Component",
        yaxis_title="Risk Contribution (bps)",
        height=400,
        showlegend=False
    )
    
    # Add informational note
    if not (portfolio_decomp or benchmark_decomp or active_decomp):
        fig.add_annotation(
            text=note_text,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=2
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Interpretation Guide:**
    
    - **Positive Allocation**: Overweighting outperforming sectors
    - **Negative Allocation**: Overweighting underperforming sectors
    - **Positive Selection**: Picking outperforming securities within sectors
    - **Negative Selection**: Picking underperforming securities within sectors
    - **Interaction**: Amplification/dampening when allocation and selection align
    """)

def render_placeholder_message():
    """Render placeholder message for unavailable features"""
    
    st.warning("""
    **Decomposition Analysis - Coming Soon**
    
    This advanced attribution analysis requires:
    - Historical return data for portfolio and benchmark
    - Sector/component mappings
    - Performance attribution calculations
    
    The framework is ready - data integration is the next step.
    """)