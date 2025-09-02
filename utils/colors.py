# Import color palette from spark-ui
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

try:
    from spark.ui.colors import COLOR_PALETTE, PLOTLY_CONTINUOUS_COLORSCALE, PLOTLY_CONTINUOUS_COLORSCALE_BLUE_WHITE_RED
except ImportError:
    # Fallback color palette if import fails
    COLOR_PALETTE = {
        "blue": "#00547b",
        "copper": "#af826f", 
        "turquoise": "#05868a",
        "green": "#b0be7d",
        "gray": "#4b4847",
        "yellow": "#d9aa60",
        "red": "#d1544c"
    }
    
    PLOTLY_CONTINUOUS_COLORSCALE = [
        [0.0, COLOR_PALETTE["blue"]],
        [0.5, COLOR_PALETTE["turquoise"]],
        [1.0, COLOR_PALETTE["red"]],
    ]

# Create factor-specific color mapping
FACTOR_COLORS = {
    "Market": COLOR_PALETTE["blue"],
    "Size": COLOR_PALETTE["turquoise"], 
    "Value": COLOR_PALETTE["green"],
    "Momentum": COLOR_PALETTE["yellow"],
    "Quality": COLOR_PALETTE["copper"],
    "Low_Vol": COLOR_PALETTE["gray"],
    "US_Region": COLOR_PALETTE["red"],
    "Europe_Region": COLOR_PALETTE["blue"],
    "EM_Region": COLOR_PALETTE["turquoise"]
}

# Standard chart colors
CHART_COLORS = {
    "portfolio": COLOR_PALETTE["blue"],
    "benchmark": COLOR_PALETTE["gray"], 
    "active": COLOR_PALETTE["red"],
    "factor_risk": COLOR_PALETTE["turquoise"],
    "specific_risk": COLOR_PALETTE["copper"],
    "positive": COLOR_PALETTE["green"],
    "negative": COLOR_PALETTE["red"],
    "neutral": COLOR_PALETTE["gray"]
}

def get_factor_color(factor_name: str) -> str:
    """Get color for a specific factor, with fallback"""
    return FACTOR_COLORS.get(factor_name, COLOR_PALETTE["gray"])

def get_chart_color(element_type: str) -> str:
    """Get color for chart elements"""
    return CHART_COLORS.get(element_type, COLOR_PALETTE["gray"])

def get_discrete_color_sequence(n_colors: int = None) -> list:
    """Get discrete color sequence for multi-series charts"""
    base_colors = list(COLOR_PALETTE.values())
    if n_colors and n_colors <= len(base_colors):
        return base_colors[:n_colors]
    return base_colors