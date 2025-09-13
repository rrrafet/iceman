import plotly.express as px
import plotly.graph_objects as go

from topgun.colors import PLOTLY_CONTINUOUS_COLORSCALE, PLOTLY_CONTINUOUS_COLORSCALE_COPPER_GRAY_GREEN, PLOTLY_CONTINUOUS_COLORSCALE_BLUE_WHITE_RED
from topgun.risk.decomposer import RiskDecomposer


class RiskDecompositionVisualizer:
    """
    Visualizes the outputs of a RiskDecomposer using Plotly Express.
    """

    def __init__(self, decomposer: RiskDecomposer, asset_names=None, factor_names=None):
        self.decomposer = decomposer
        self.asset_names = asset_names
        self.factor_names = factor_names

    def asset_contributions(self, kind="absolute", fmt='.1%', width=800, height=500, **kwargs):
        """
        Visualize asset contributions to risk.
        kind: "percent" for percent contribution, "absolute" for absolute contribution.
        fmt: Format string for y-axis labels (default '.1%').
        width: Width of the figure (default 800).
        height: Height of the figure (default 500).
        """
        if kind == "absolute":
            values = self.decomposer.ctr
        else:
            values = self.decomposer.pctr
        
        title = "Contribution to Risk by Asset"
        yaxis = "Contribution"

        # Use only the first token of each symbol from model if available
        if hasattr(self.decomposer.model, "symbols"):
            names = [s.split()[0] for s in self.decomposer.model.symbols]
        else:
            names = self.asset_names or [f"Asset {i}" for i in range(len(values))]

        fig = px.bar(
            x=names,
            y=values,
            labels={"x": "Asset", "y": yaxis},
            title=title,
            width=width,
            height=height,
            **kwargs
        )
        fig.update_traces(text=[format(v if fmt.endswith('%') else v, fmt) for v in values], textposition='outside')
        fig.update_yaxes(tickformat=fmt)
        fig.show()

    def factor_contributions(self, kind="absolute", fmt='.1%', width=800, height=500, **kwargs):
        """
        Visualize factor contributions to risk.
        kind: "percent" for percent contribution, "absolute" for absolute contribution.
        fmt: Format string for y-axis labels (default '.1%').
        width: Width of the figure (default 800).
        height: Height of the figure (default 500).
        """
        if kind == "percent":
            values = self.decomposer.pfctr
        else:
            values = self.decomposer.fctr
        
        title = "Contribution to Risk by Factor"
        yaxis = "Contribution"

        # Use only the first token of each factor name from model if available
        if hasattr(self.decomposer.model, "factor_names"):
            names = [s.split()[0] for s in self.decomposer.model.factor_names]
        else:
            names = self.factor_names or [f"Factor {i}" for i in range(len(values))]

        fig = px.bar(
            x=names,
            y=values,
            labels={"x": "Factor", "y": yaxis},
            title=title,
            width=width,
            height=height,
            **kwargs
        )
        fig.update_traces(text=[format(v if fmt.endswith('%') else v, fmt) for v in values], textposition='outside')
        fig.update_yaxes(tickformat=fmt)
        fig.show()
        return fig

    def asset_factor_heatmap(self, fmt='.1%', width=800, height=500, **kwargs):
        """
        Visualize a heatmap of asset-factor contributions to risk.
        fmt: Format string for cell labels (default '.1%').
        width: Width of the figure (default 800).
        height: Height of the figure (default 500).
        """
        data = self.decomposer.ctr_asset_factor

        # Use only the first token of each symbol and factor name from model if available
        if hasattr(self.decomposer.model, "symbols"):
            asset_names = [s.split()[0] for s in self.decomposer.model.symbols]
        else:
            asset_names = self.asset_names or [f"Asset {i}" for i in range(data.shape[0])]

        if hasattr(self.decomposer.model, "factor_names"):
            factor_names = [s.split()[0] for s in self.decomposer.model.factor_names]
        else:
            factor_names = self.factor_names or [f"Factor {i}" for i in range(data.shape[1])]

        fig = px.imshow(
            data.T,
            y=factor_names,
            x=asset_names,
            labels={"x": "", "y": "", "color": "Contribution"},
            title="Asset-Factor Contribution to Risk",
            text_auto=fmt,
            color_continuous_scale=PLOTLY_CONTINUOUS_COLORSCALE_COPPER_GRAY_GREEN,
            color_continuous_midpoint=0,
            width=width,
            height=height,
            **kwargs
        )
        # hide the color bar
        fig.update_coloraxes(colorbar_tickformat=fmt, showscale=False)
        fig.show()
        return fig

    def factor_exposures(self, fmt='.1f', width=800, height=500, **kwargs):
        """
        Visualize portfolio-level exposures to each factor.
        fmt: Format string for y-axis labels (default '.1f').
        width: Width of the figure (default 800).
        height: Height of the figure (default 500).
        """
        exposures = self.decomposer.factor_exposure

        # Use only the first token of each factor name from model if available
        if hasattr(self.decomposer.model, "factor_names"):
            names = [s.split()[0] for s in self.decomposer.model.factor_names]
        else:
            names = self.factor_names or [f"Factor {i}" for i in range(len(exposures))]

        fig = px.bar(
            x=names,
            y=exposures,
            labels={"y": "Beta", "x": "Factor"},
            title="Portfolio Factor Exposures",
            width=width,
            height=height,
            **kwargs
        )
        fig.update_traces(text=[format(v, fmt) for v in exposures], textposition='outside')
        fig.update_yaxes(tickformat=fmt)
        fig.show()
        return fig

    def betas(self, fmt='.2f', width=800, height=500, **kwargs):
        """
        Visualize a heatmap of asset-factor betas.
        fmt: Format string for cell labels (default '.2f').
        width: Width of the figure (default 800).
        height: Height of the figure (default 500).
        """
        # Assume the decomposer has an attribute 'betas' (shape: assets x factors)
        data = self.decomposer.beta

        # Use only the first token of each symbol and factor name from model if available
        if hasattr(self.decomposer.model, "symbols"):
            asset_names = [s.split()[0] for s in self.decomposer.model.symbols]
        else:
            asset_names = self.asset_names or [f"Asset {i}" for i in range(data.shape[0])]

        if hasattr(self.decomposer.model, "factor_names"):
            factor_names = [s.split()[0] for s in self.decomposer.model.factor_names]
        else:
            factor_names = self.factor_names or [f"Factor {i}" for i in range(data.shape[1])]

        fig = px.imshow(
            data.T,
            y=factor_names,
            x=asset_names,
            labels={"y": "", "x": "", "color": "Beta"},
            title="Asset-Factor Betas",
            text_auto=fmt,
            color_continuous_scale=PLOTLY_CONTINUOUS_COLORSCALE_COPPER_GRAY_GREEN,
            color_continuous_midpoint=0,
            width=width,
            height=height,
            **kwargs
        )
        # hide the color bar
        fig.update_coloraxes(colorbar_tickformat=fmt, showscale=False)
        fig.show()
        return fig

    def totals(self, fmt='.1%', width=800, height=500, **kwargs):
        """
        Visualize a bar chart of total risk contributions:
        - Total (ctr.sum())
        - Factor (ctr_factor.sum())
        - Idiosyncratic (ctr_idio.sum())
        fmt: Format string for y-axis labels (default '.1%').
        width: Width of the figure (default 800).
        height: Height of the figure (default 500).
        """
        total = self.decomposer.ctr.sum()
        factor = self.decomposer.ctr_factor.sum()
        idio = self.decomposer.ctr_idio.sum()
        values = [total, factor, idio]
        names = ["Total", "Factor", "Idiosyncratic"]

        fig = px.bar(
            x=names,
            y=values,
            labels={"x": "Contribution Type", "y": "Contribution"},
            title="Total Risk Contributions",
            width=width,
            height=height,
            **kwargs
        )
        fig.update_traces(text=[format(v, fmt) for v in values], textposition='outside')
        fig.update_yaxes(tickformat=fmt)
        fig.show()