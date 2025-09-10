"""
Stats Tab Component - Hierarchical statistics view for data reconciliation.

This component displays a hierarchical table of portfolio statistics showing:
- Portfolio/benchmark/active weights relative to root
- Computed statistics from PortfolioGraph
- Raw time series statistics
- Reconciliation indicators
"""

import pandas as pd
import streamlit as st
import numpy as np
from typing import Dict, Any, List, Optional


def render_stats_tab(data_access_service, sidebar_state):
    """
    Render the Stats tab showing hierarchical statistics for data reconciliation.
    
    Args:
        data_access_service: Data access service instance
        sidebar_state: Sidebar state containing selected filters
    """
    st.header("Portfolio Statistics & Reconciliation")
    
    # Add description
    st.markdown("""
    This view shows comprehensive statistics for all components in the portfolio hierarchy,
    helping reconcile that data inputs are correctly transformed through the risk analysis system.
    
    **Weight Sources for Raw Volatility Calculations:**
    - **Regular Components**: Use weights relative to portfolio root for calculations
    - **Overlay Strategies** (🔄): Use operational weights fixed at 1.0 for portfolio risk, 0.0 for benchmark risk
    - **Allocation vs Operational**: Overlays have 0.0% allocation weights (prevent double-counting) but 1.0 operational weights (enable risk calculations)
    """)
    
    # Controls row
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        show_computed = st.checkbox("Show Computed Stats", value=True, 
                                   help="Display statistics from risk computation engine using PortfolioGraph")
    
    with col2:
        show_raw = st.checkbox("Show Raw Stats", value=True,
                              help="Display statistics from raw time series data. For overlays, uses operational weights (1.0) for portfolio risk calculations.")
    
    with col3:
        highlight_discrepancies = st.checkbox("Highlight Discrepancies", value=True,
                                             help="Highlight differences between computed and raw stats. Overlays may show expected differences due to operational weight handling.")
    
    # Get hierarchical statistics
    with st.spinner("Loading hierarchical statistics..."):
        try:
            stats_data = data_access_service.get_hierarchical_stats()
            
            if not stats_data:
                st.warning("No statistics data available")
                return
            
            # Render the hierarchical table
            render_hierarchical_stats_table(
                stats_data, 
                show_computed=show_computed,
                show_raw=show_raw,
                highlight_discrepancies=highlight_discrepancies
            )
            
            # Export functionality
            st.divider()
            
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("📥 Export to CSV"):
                    df = stats_to_dataframe(stats_data, show_computed, show_raw)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="portfolio_stats.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.caption("Export the statistics table for further analysis. Includes overlay flags and weight source information.")
            
        except Exception as e:
            st.error(f"Error loading statistics: {e}")


def render_hierarchical_stats_table(
    stats_data: List[Dict[str, Any]], 
    show_computed: bool = True,
    show_raw: bool = True,
    highlight_discrepancies: bool = True
):
    """
    Render the hierarchical statistics table with proper formatting.
    
    Args:
        stats_data: List of component statistics in hierarchical order
        show_computed: Whether to show computed statistics columns
        show_raw: Whether to show raw time series statistics columns
        highlight_discrepancies: Whether to highlight discrepancies
    """
    if not stats_data:
        st.info("No data to display")
        return
    
    # Build the display data
    table_data = []
    
    for stats in stats_data:
        component_id = stats["component_id"]
        level = stats.get("level", 0)
        is_leaf = stats.get("is_leaf", True)
        is_overlay = stats.get("is_overlay", False)
        weights = stats["weights"]
        weight_source = stats.get("weight_source", {})
        computed_stats = stats["computed_stats"]
        raw_stats = stats["raw_stats"]
        
        # Create indented component name with overlay indicator
        indent = "  " * level
        if is_overlay:
            icon = "🔄" if is_leaf else "📁🔄"  # Overlay indicator
            component_type = "Overlay" if is_leaf else "Node (Overlay)"
        else:
            icon = "📄" if is_leaf else "📁"
            component_type = "Leaf" if is_leaf else "Node"
        display_name = f"{indent}{icon} {component_id}"
        
        # Build row data
        row = {
            "Component": display_name,
            "Type": component_type,
            "Is Overlay": "✅" if is_overlay else "—",
            "Portfolio Weight": f"{weights['portfolio']:.2%}" if not np.isnan(weights['portfolio']) else "—",
            "Benchmark Weight": f"{weights['benchmark']:.2%}" if not np.isnan(weights['benchmark']) else "—",
            "Active Weight": f"{weights['active']:.2%}" if not np.isnan(weights['active']) else "—",
            "Weight Source": weight_source.get("description", "Root Relative") if weight_source else "Root Relative",
        }
        
        # Add computed statistics if requested
        if show_computed:
            row.update({
                "Computed Port Vol": format_stat(computed_stats["portfolio"]["std"]),
                "Computed Bench Vol": format_stat(computed_stats["benchmark"]["std"]),
                "Computed Active Vol": format_stat(computed_stats["active"]["std"]),
            })
        
        # Add raw statistics if requested
        if show_raw:
            row.update({
                "Raw Port Mean": format_stat(raw_stats["portfolio"]["mean"]),
                "Raw Port Vol": format_stat(raw_stats["portfolio"]["std"]),
                "Raw Bench Mean": format_stat(raw_stats["benchmark"]["mean"]),
                "Raw Bench Vol": format_stat(raw_stats["benchmark"]["std"]),
                "Raw Active Mean": format_stat(raw_stats["active"]["mean"]),
                "Raw Active Vol": format_stat(raw_stats["active"]["std"]),
            })
        
        # Add reconciliation status if both computed and raw are shown
        if show_computed and show_raw and highlight_discrepancies:
            # Check for discrepancies in volatility (standard deviation)
            port_match = check_stat_match(
                computed_stats["portfolio"]["std"], 
                raw_stats["portfolio"]["std"]
            )
            bench_match = check_stat_match(
                computed_stats["benchmark"]["std"],
                raw_stats["benchmark"]["std"]
            )
            active_match = check_stat_match(
                computed_stats["active"]["std"],
                raw_stats["active"]["std"]
            )
            
            if port_match and bench_match and active_match:
                row["Status"] = "✅ Match"
            else:
                mismatches = []
                if not port_match:
                    mismatches.append("Port")
                if not bench_match:
                    mismatches.append("Bench")
                if not active_match:
                    mismatches.append("Active")
                row["Status"] = f"⚠️ Mismatch ({', '.join(mismatches)})"
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Apply styling if highlighting discrepancies
    if highlight_discrepancies and "Status" in df.columns:
        # Use Streamlit's dataframe with custom styling
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Component": st.column_config.TextColumn(
                    "Component",
                    width="large",
                ),
                "Type": st.column_config.TextColumn(
                    "Type",
                    help="Component type: Leaf/Node components or Overlay strategies",
                    width="small",
                ),
                "Is Overlay": st.column_config.TextColumn(
                    "Is Overlay",
                    help="✅ = Overlay strategy using operational weights (1.0) for risk calculations",
                    width="small",
                ),
                "Weight Source": st.column_config.TextColumn(
                    "Weight Source", 
                    help="Source of weights used for raw volatility calculations",
                    width="medium",
                ),
                "Status": st.column_config.TextColumn(
                    "Status",
                    width="small",
                ),
            }
        )
    else:
        # Regular dataframe display with column config
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Component": st.column_config.TextColumn(
                    "Component",
                    width="large",
                ),
                "Type": st.column_config.TextColumn(
                    "Type",
                    help="Component type: Leaf/Node components or Overlay strategies",
                    width="small",
                ),
                "Is Overlay": st.column_config.TextColumn(
                    "Is Overlay",
                    help="✅ = Overlay strategy using operational weights (1.0) for risk calculations",
                    width="small",
                ),
                "Weight Source": st.column_config.TextColumn(
                    "Weight Source", 
                    help="Source of weights used for raw volatility calculations",
                    width="medium",
                ),
            }
        )
    
    # Summary statistics
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Components", len(stats_data))
    
    with col2:
        leaf_count = sum(1 for s in stats_data if s["is_leaf"])
        st.metric("Leaf Components", leaf_count)
    
    with col3:
        node_count = len(stats_data) - leaf_count
        st.metric("Node Components", node_count)
    
    with col4:
        if "Status" in df.columns:
            match_count = sum(1 for _, row in df.iterrows() if "Match" in str(row.get("Status", "")))
            match_pct = (match_count / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Reconciliation Rate", f"{match_pct:.1f}%")


def format_stat(value: float, precision: int = 2) -> str:
    """
    Format a statistical value for display.
    
    Args:
        value: The value to format
        precision: Number of decimal places for percentage
        
    Returns:
        Formatted string
    """
    if np.isnan(value) or value is None:
        return "—"
    
    # Convert to percentage (assuming annualized decimal values)
    return f"{value:.{precision}%}"


def check_stat_match(computed: float, raw: float, tolerance: float = 0.001) -> bool:
    """
    Check if computed and raw statistics match within tolerance.
    
    Args:
        computed: Computed statistic value
        raw: Raw statistic value
        tolerance: Tolerance for comparison (default 0.1%)
        
    Returns:
        True if values match within tolerance
    """
    # Handle NaN cases
    if np.isnan(computed) and np.isnan(raw):
        return True
    if np.isnan(computed) or np.isnan(raw):
        return False
    
    # Check if both are effectively zero
    if abs(computed) < tolerance and abs(raw) < tolerance:
        return True
    
    # Check relative difference
    if computed != 0:
        relative_diff = abs((raw - computed) / computed)
        return relative_diff < tolerance
    
    return False


def stats_to_dataframe(
    stats_data: List[Dict[str, Any]], 
    include_computed: bool = True,
    include_raw: bool = True
) -> pd.DataFrame:
    """
    Convert hierarchical statistics to a flat DataFrame for export.
    
    Args:
        stats_data: List of component statistics
        include_computed: Whether to include computed statistics
        include_raw: Whether to include raw statistics
        
    Returns:
        DataFrame suitable for export
    """
    rows = []
    
    for stats in stats_data:
        weight_source = stats.get("weight_source", {})
        row = {
            "component_id": stats["component_id"],
            "level": stats.get("level", 0),
            "is_leaf": stats.get("is_leaf", True),
            "is_overlay": stats.get("is_overlay", False),
            "portfolio_weight": stats["weights"]["portfolio"],
            "benchmark_weight": stats["weights"]["benchmark"],
            "active_weight": stats["weights"]["active"],
            "portfolio_weight_source": weight_source.get("portfolio", "Root Relative"),
            "benchmark_weight_source": weight_source.get("benchmark", "Root Relative"),
            "active_weight_source": weight_source.get("active", "Root Relative"),
            "weight_source_description": weight_source.get("description", "Root Relative"),
        }
        
        if include_computed:
            row.update({
                "computed_portfolio_mean": stats["computed_stats"]["portfolio"]["mean"],
                "computed_portfolio_std": stats["computed_stats"]["portfolio"]["std"],
                "computed_benchmark_mean": stats["computed_stats"]["benchmark"]["mean"],
                "computed_benchmark_std": stats["computed_stats"]["benchmark"]["std"],
                "computed_active_mean": stats["computed_stats"]["active"]["mean"],
                "computed_active_std": stats["computed_stats"]["active"]["std"],
            })
        
        if include_raw:
            row.update({
                "raw_portfolio_mean": stats["raw_stats"]["portfolio"]["mean"],
                "raw_portfolio_std": stats["raw_stats"]["portfolio"]["std"],
                "raw_benchmark_mean": stats["raw_stats"]["benchmark"]["mean"],
                "raw_benchmark_std": stats["raw_stats"]["benchmark"]["std"],
                "raw_active_mean": stats["raw_stats"]["active"]["mean"],
                "raw_active_std": stats["raw_stats"]["active"]["std"],
            })
        
        rows.append(row)
    
    return pd.DataFrame(rows)