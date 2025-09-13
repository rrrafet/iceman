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

PLOTLY_CONTINUOUS_COLORSCALE_COPPER_GRAY_GREEN = [
    [0.0, COLOR_PALETTE["copper"]],
    [0.0, COLOR_PALETTE["gray"]],
    [1.0, COLOR_PALETTE["green"]],
]

PLOTLY_CONTINUOUS_COLORSCALE_BLUE_WHITE_RED = [
    [0.0, COLOR_PALETTE["blue"]],
    [0.5, "#ffffff"],  # white
    [1.0, COLOR_PALETTE["red"]],
]