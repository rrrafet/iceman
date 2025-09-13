"""
Utility functions for data formatting and display
"""
from typing import Union, List, Dict, Any

from spark.risk.annualizer import RiskAnnualizer

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format value as percentage with specified decimal places"""
    return f"{value:.{decimal_places}f}%"

def format_basis_points(value: float, decimal_places: int = 0) -> str:
    """Format value as basis points"""
    return f"{value * 10000:.{decimal_places}f} bps"

def format_decimal(value: float, decimal_places: int = 4) -> str:
    """Format value as decimal with specified places"""
    return f"{value:.{decimal_places}f}"

def format_currency(value: float, currency: str = "USD", decimal_places: int = 2) -> str:
    """Format value as currency"""
    return f"{currency} {value:,.{decimal_places}f}"

def format_risk_metric(value: float, metric_type: str = "basis_points") -> str:
    """Format risk metrics based on type"""
    if metric_type == "percentage":
        return format_percentage(value)
    elif metric_type == "basis_points":
        return format_basis_points(value)
    elif metric_type == "decimal":
        return format_decimal(value)
    else:
        return f"{value:.2f}"

def truncate_component_name(name: str, max_length: int = 15) -> str:
    """Truncate long component names for display"""
    if len(name) <= max_length:
        return name
    return name[:max_length-3] + "..."

def format_large_number(value: float, suffix: str = "") -> str:
    """Format large numbers with appropriate suffixes"""
    if abs(value) >= 1e9:
        return f"{value/1e9:.1f}B{suffix}"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.1f}M{suffix}"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.1f}K{suffix}"
    else:
        return f"{value:.1f}{suffix}"

def get_risk_level_badge(risk_value: float, thresholds: Dict[str, float] = None) -> tuple:
    """Return badge color and text for risk level"""
    if thresholds is None:
        thresholds = {"low": 100, "medium": 300, "high": 500}  # basis points
    
    risk_bps = abs(risk_value * 10000)  # Convert to basis points
    
    if risk_bps <= thresholds["low"]:
        return "success", "Low Risk"
    elif risk_bps <= thresholds["medium"]:
        return "warning", "Medium Risk" 
    else:
        return "error", "High Risk"

def format_date_range_label(start_idx: int, end_idx: int, total_periods: int, frequency: str = "D") -> str:
    """Format date range for display"""
    periods = end_idx - start_idx + 1
    if start_idx == 0 and end_idx == total_periods - 1:
        return "Full Range"
    
    periods_per_year = RiskAnnualizer.get_periods_per_year(frequency)
    one_year_periods = int(periods_per_year)
    three_year_periods = int(3 * periods_per_year)
    
    if periods <= one_year_periods:  # Roughly 1 year
        return f"Last {periods} periods"
    elif periods <= three_year_periods:  # Roughly 3 years
        return f"Last ~{periods//one_year_periods} years"
    else:
        return f"{periods} periods"

def clean_factor_name(factor_name: str) -> str:
    """Clean factor names for display"""
    # Replace underscores with spaces and title case
    return factor_name.replace("_", " ").title()

def format_correlation(value: float) -> str:
    """Format correlation values with appropriate precision"""
    return f"{value:.3f}"

def format_p_value(p_value: float) -> str:
    """Format p-values with significance indicators"""
    if p_value < 0.001:
        return f"{p_value:.4f}***"
    elif p_value < 0.01:
        return f"{p_value:.4f}**"
    elif p_value < 0.05:
        return f"{p_value:.4f}*"
    else:
        return f"{p_value:.4f}"

def get_significance_stars(p_value: float) -> str:
    """Get significance stars for p-values"""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""

def format_weight(weight: float, as_percentage: bool = True) -> str:
    """Format portfolio weights"""
    if as_percentage:
        return f"{weight:.2f}%"
    else:
        return f"{weight:.4f}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default

def highlight_extreme_values(values: List[float], threshold_percentile: float = 95) -> List[bool]:
    """Identify extreme values in a list (beyond threshold percentile)"""
    if not values:
        return []
    
    import numpy as np
    threshold = np.percentile(np.abs(values), threshold_percentile)
    return [abs(v) >= threshold for v in values]