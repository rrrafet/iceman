import plotly.io as pio
import plotly.graph_objects as go
from .colors import COLOR_PALETTE

def register_plotly_template():
    custom_colors = list(COLOR_PALETTE.values())

    font_family = "Corbel, Segoe UI, Verdana, Arial, sans-serif"
    title_color = "#000000"  # black

    custom_template = go.layout.Template(
        layout=go.Layout(
            colorway=custom_colors,
            font=dict(family=font_family, size=12, color=COLOR_PALETTE["gray"]),
            title=dict(font=dict(family=font_family, size=18, color=title_color)),
            xaxis=dict(
                title_font=dict(family=font_family, size=14, color=title_color),
                tickfont=dict(size=14),
                showgrid=True,
                zeroline=False,
                showline=True,
                linecolor=COLOR_PALETTE["gray"],
                gridcolor='rgba(0, 0, 0, 0.05)',
                ticks='outside'
            ),
            yaxis=dict(
                title_font=dict(family=font_family, size=14, color=title_color),
                tickfont=dict(size=14),
                showgrid=True,
                zeroline=False,
                showline=True,
                linecolor=COLOR_PALETTE["gray"],
                gridcolor='rgba(0, 0, 0, 0.05)',
                ticks='outside'
            ),
            legend=dict(
                font=dict(family=font_family, size=12, color=COLOR_PALETTE["gray"]),
                title=dict(font=dict(family=font_family, size=14, color=title_color)),
                orientation="h",
                x=0,
                y=1.1
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
    )

    pio.templates["AP1"] = custom_template
    pio.templates.default = "AP1"