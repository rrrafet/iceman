"""
Reconciliation Tab - Volatility Validation of Risk Calculation Inputs.

This component validates that volatilities calculated from the exact filtered and
resampled data that goes into risk calculations match the computed risk values:
- Uses the same DataAccessService methods that feed risk calculations
- Compares computed volatilities from risk engine vs empirical from processed data
- Ensures data consistency between risk inputs and risk outputs
- Validates across all lenses (portfolio, benchmark, active) and frequencies
"""

import pandas as pd
import streamlit as st
import numpy as np
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


def render_reconciliation_tab(data_access_service, sidebar_state):
    """
    Render volatility reconciliation comparing risk calculation inputs vs outputs.
    
    Args:
        data_access_service: Data access service with ResamplingService integration
        sidebar_state: Sidebar state with frequency and date range settings
    """
    st.header("Volatility Reconciliation")
    st.caption("Validates that computed risk values match empirical volatilities from the same filtered/resampled data used in risk calculations")
    
    # Show current data processing settings
    with st.expander("Data Processing Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Frequency", sidebar_state.frequency)
        with col2:
            try:
                freq_status = data_access_service.get_frequency_status()
                resampling_status = "Resampled" if freq_status.get('is_resampled', False) else "Native"
                st.metric("Processing", resampling_status)
            except Exception as e:
                logger.warning(f"Could not get frequency status: {e}")
                st.metric("Processing", "Unknown")
        with col3:
            if sidebar_state.date_range_start and sidebar_state.date_range_end:
                days = (sidebar_state.date_range_end - sidebar_state.date_range_start).days
                st.metric("Period", f"{days/365:.1f} years")
    
    # Validation controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("**Validation Process**: Compares computed volatilities from risk engine against empirical volatilities calculated from the exact same filtered/resampled return data that feeds into risk calculations")
    
    with col2:
        tolerance_basis_points = st.number_input(
            "Tolerance (basis points)", 
            min_value=1, 
            max_value=100, 
            value=10,
            help="Tolerance for volatility matching (10 bp = 0.1%)"
        )
        tolerance = tolerance_basis_points / 10000  # Convert bp to decimal
    
    # Get and validate hierarchical statistics
    with st.spinner("Loading volatility statistics from filtered/resampled data..."):
        try:
            stats_data = data_access_service.get_hierarchical_stats()
            
            if not stats_data:
                st.warning("No statistics data available - check data loading and date range settings")
                return
            
            # Show processing summary
            st.success(f"Loaded {len(stats_data)} components for volatility validation")
            
            # Render the volatility comparison table
            render_volatility_validation_table(stats_data, tolerance)
            
            # Export functionality
            st.divider()
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Export Validation Results"):
                    df = create_validation_export(stats_data, tolerance, sidebar_state.frequency)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"volatility_validation_{sidebar_state.frequency}_{tolerance_basis_points}bp.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.caption("Export detailed volatility validation results including differences and match status")
                
        except Exception as e:
            st.error(f"Error loading volatility statistics: {e}")
            logger.error(f"Reconciliation tab error: {e}")
            
            # Show troubleshooting info
            with st.expander("Troubleshooting", expanded=False):
                st.write("**Common issues:**")
                st.write("- Data providers not properly initialized with frequency/date settings")
                st.write("- Selected date range has insufficient data for the chosen frequency") 
                st.write("- Risk computation service not available or failed initialization")
                st.write("- Mismatch between sidebar settings and data service state")
                
                # Show service status if possible
                try:
                    status = data_access_service.get_service_status()
                    st.json(status)
                except:
                    st.write("Could not retrieve service status for debugging")


def render_volatility_validation_table(
    stats_data: List[Dict[str, Any]], 
    tolerance: float = 0.001
):
    """
    Render table comparing computed vs empirical volatilities from risk calculation inputs.
    
    Args:
        stats_data: List of component statistics in hierarchical order
        tolerance: Tolerance for volatility matching (decimal)
    """
    if not stats_data:
        st.warning("No data available for volatility validation")
        return
    
    # Build volatility comparison data
    table_data = []
    match_counts = {"portfolio": 0, "benchmark": 0, "active": 0}
    total_counts = {"portfolio": 0, "benchmark": 0, "active": 0}
    
    for stats in stats_data:
        component_id = stats["component_id"]
        level = stats.get("level", 0)
        is_leaf = stats.get("is_leaf", True)
        is_overlay = stats.get("is_overlay", False)
        weights = stats["weights"]
        computed_stats = stats["computed_stats"]
        raw_stats = stats["raw_stats"]
        
        # Create indented component name
        indent = "  " * level
        component_type = "Overlay" if is_overlay else ("Leaf" if is_leaf else "Node")
        display_name = f"{indent}{component_id}"
        
        # Build row focusing on volatility validation
        row = {
            "Component": display_name,
            "Type": component_type,
            "Portfolio Weight": format_weight(weights["portfolio"]),
            "Benchmark Weight": format_weight(weights["benchmark"]), 
            "Active Weight": format_weight(weights["active"]),
        }
        
        # Add volatility comparisons and validation for each lens
        lenses = ["portfolio", "benchmark", "active"]
        lens_labels = {"portfolio": "Port", "benchmark": "Bench", "active": "Active"}
        
        for lens in lenses:
            computed_vol = computed_stats[lens]["std"]
            empirical_vol = raw_stats[lens]["std"]
            
            # Format volatilities
            computed_str = format_volatility(computed_vol)
            empirical_str = format_volatility(empirical_vol)
            
            # Validate match and track statistics
            matches = validate_volatility_match(computed_vol, empirical_vol, tolerance)
            
            # Track validation statistics
            if not (np.isnan(computed_vol) and np.isnan(empirical_vol)):
                total_counts[lens] += 1
                if matches:
                    match_counts[lens] += 1
            
            # Calculate difference for display
            if not np.isnan(computed_vol) and not np.isnan(empirical_vol):
                diff_bp = (empirical_vol - computed_vol) * 10000
                diff_str = f"{diff_bp:+.1f}bp"
                match_str = "MATCH" if matches else "DIFF"
            else:
                diff_str = "N/A"
                match_str = "N/A"
            
            # Add columns for this lens
            row[f"{lens_labels[lens]} Computed Vol"] = computed_str
            row[f"{lens_labels[lens]} Empirical Vol"] = empirical_str
            row[f"{lens_labels[lens]} Difference"] = diff_str
            row[f"{lens_labels[lens]} Status"] = match_str
        
        table_data.append(row)
    
    # Create and display DataFrame
    df = pd.DataFrame(table_data)
    
    if df.empty:
        st.warning("No data available for volatility comparison")
        return
    
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "Component": st.column_config.TextColumn(
                "Component",
                width="medium",
                help="Portfolio components in hierarchical structure"
            ),
            "Type": st.column_config.TextColumn(
                "Type", 
                width="small",
                help="Leaf components, Node aggregations, or Overlay strategies"
            ),
        }
    )
    
    # Validation Summary
    st.divider()
    st.subheader("Validation Summary")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Components", len(stats_data))
    
    with col2:
        overlay_count = sum(1 for s in stats_data if s.get("is_overlay", False))
        st.metric("Overlays", overlay_count)
    
    with col3:
        total_validations = sum(total_counts.values())
        st.metric("Total Validations", total_validations)
    
    with col4:
        if total_validations > 0:
            total_matches = sum(match_counts.values()) 
            overall_match_pct = (total_matches / total_validations) * 100
            st.metric("Overall Match Rate", f"{overall_match_pct:.1f}%")
        else:
            st.metric("Overall Match Rate", "No data")
    
    # Lens-specific validation rates
    st.subheader("Validation by Lens")
    col1, col2, col3 = st.columns(3)
    
    for i, (lens, label) in enumerate(zip(["portfolio", "benchmark", "active"], 
                                         ["Portfolio", "Benchmark", "Active"])):
        with [col1, col2, col3][i]:
            if total_counts[lens] > 0:
                match_pct = (match_counts[lens] / total_counts[lens]) * 100
                st.metric(
                    f"{label} Match Rate",
                    f"{match_pct:.1f}%",
                    help=f"{match_counts[lens]} of {total_counts[lens]} components match within tolerance"
                )
                
                if match_pct < 90:
                    st.error(f"Low validation rate for {lens} lens")
                elif match_pct < 100:
                    st.warning(f"Some {lens} components don't validate")
                else:
                    st.success(f"Perfect {lens} validation")
            else:
                st.metric(f"{label} Match Rate", "No data")


def format_weight(value: float) -> str:
    """Format weight value for display."""
    if np.isnan(value) or value is None:
        return "N/A"
    return f"{value:.2%}"


def format_volatility(value: float, precision: int = 1) -> str:
    """Format volatility value for display."""
    if np.isnan(value) or value is None:
        return "N/A"
    return f"{value:.{precision}%}"


def validate_volatility_match(computed: float, empirical: float, tolerance: float = 0.001) -> bool:
    """
    Validate that computed and empirical volatilities match within tolerance.
    
    This ensures that the risk calculation inputs (empirical volatilities from 
    filtered/resampled data) match the risk calculation outputs (computed volatilities).
    
    Args:
        computed: Computed volatility from risk engine
        empirical: Empirical volatility from same filtered/resampled data
        tolerance: Tolerance for comparison (decimal, e.g. 0.001 = 10bp)
        
    Returns:
        True if volatilities match within tolerance
    """
    # Handle NaN cases
    if np.isnan(computed) and np.isnan(empirical):
        return True
    if np.isnan(computed) or np.isnan(empirical):
        return False
    
    # Check if both are effectively zero (very low volatility)
    if abs(computed) < tolerance/10 and abs(empirical) < tolerance/10:
        return True
    
    # Check absolute difference for volatilities
    absolute_diff = abs(empirical - computed)
    return absolute_diff <= tolerance


def create_validation_export(
    stats_data: List[Dict[str, Any]], 
    tolerance: float,
    frequency: str
) -> pd.DataFrame:
    """
    Create detailed DataFrame for exporting validation results.
    
    Args:
        stats_data: List of component statistics
        tolerance: Tolerance used for validation
        frequency: Current data frequency
        
    Returns:
        DataFrame with comprehensive validation results
    """
    rows = []
    
    for stats in stats_data:
        component_id = stats["component_id"]
        level = stats.get("level", 0)
        is_leaf = stats.get("is_leaf", True) 
        is_overlay = stats.get("is_overlay", False)
        weights = stats["weights"]
        computed_stats = stats["computed_stats"]
        raw_stats = stats["raw_stats"]
        
        # Base row data
        row = {
            "component_id": component_id,
            "hierarchy_level": level,
            "is_leaf": is_leaf,
            "is_overlay": is_overlay,
            "frequency": frequency,
            "tolerance_bp": tolerance * 10000,
            "portfolio_weight": weights["portfolio"],
            "benchmark_weight": weights["benchmark"],
            "active_weight": weights["active"],
        }
        
        # Add detailed volatility comparison for each lens
        for lens in ["portfolio", "benchmark", "active"]:
            computed_vol = computed_stats[lens]["std"]
            empirical_vol = raw_stats[lens]["std"]
            matches = validate_volatility_match(computed_vol, empirical_vol, tolerance)
            
            # Calculate difference in basis points
            if not np.isnan(computed_vol) and not np.isnan(empirical_vol):
                difference_bp = (empirical_vol - computed_vol) * 10000
            else:
                difference_bp = np.nan
            
            row.update({
                f"{lens}_computed_volatility": computed_vol,
                f"{lens}_empirical_volatility": empirical_vol, 
                f"{lens}_difference_bp": difference_bp,
                f"{lens}_validates": matches,
                f"{lens}_within_tolerance": abs(difference_bp) <= tolerance * 10000 if not np.isnan(difference_bp) else False
            })
        
        rows.append(row)
    
    return pd.DataFrame(rows)