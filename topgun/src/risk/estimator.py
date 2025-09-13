import warnings
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.stats.correlation_tools import cov_nearest
from typing import Optional, Tuple, List

from statsmodels.tools import add_constant
from .model import RiskModel

class LinearRiskModel(RiskModel):
    def __init__(self, beta, factor_covar, resvar, frequency, symbols, factor_names,
                 asset_returns=None, factor_returns=None, residual_returns=None):
        self._beta = beta
        self._factor_covar = factor_covar
        self._resvar = resvar
        self._frequency = frequency
        self._symbols = symbols
        self._factor_names = factor_names
        self._asset_returns = asset_returns
        self._factor_returns = factor_returns
        self._residual_returns = residual_returns

    @property
    def beta(self):
        return self._beta

    @property
    def factor_covar(self):
        return self._factor_covar

    @property
    def resvar(self):
        return self._resvar

    @property
    def covar(self):
        return (self._beta @ self._factor_covar @ self._beta.T) + self._resvar

    @property
    def frequency(self):
        return self._frequency

    @property
    def symbols(self):
        return self._symbols

    @property
    def factor_names(self):
        return self._factor_names

    @property
    def asset_returns(self):
        return self._asset_returns

    @property
    def factor_returns(self):
        return self._factor_returns

    @property
    def residual_returns(self):
        return self._residual_returns

class LinearRiskModelEstimator:
    def __init__(
        self,
        winsorize_level: Optional[float] = None,
        regression_type: str = 'ols',
        min_obs: int = 30,
        freq: Optional[str] = None
    ) -> None:
        self.regression_type = regression_type
        self.freq = freq
        self.min_obs = min_obs
        self.winsorize_level = winsorize_level

    def _check_frequency(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame
    ) -> str:
        if self.freq is not None:
            return self.freq
        
        freq1 = pd.infer_freq(df1.index)
        freq2 = pd.infer_freq(df2.index)
        if freq1 != freq2:
            raise ValueError(f"Frequency mismatch: asset_returns={freq1}, factor_returns={freq2}")
        if freq1 is None:
            raise ValueError("Cannot infer frequency from asset_returns and factor_returns.")
        return freq1

    def _winsorize(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        if self.winsorize_level is None:
            return df
        lower = self.winsorize_level / 2
        upper = 1 - lower
        return df.clip(df.quantile(lower), df.quantile(upper), axis=1)

    def fit(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> 'LinearRiskModel':
        asset_returns, factor_returns = asset_returns.align(factor_returns, join='inner', axis=0)
        freq = self._check_frequency(asset_returns, factor_returns)

        asset_returns = self._winsorize(asset_returns)
        factor_returns = self._winsorize(factor_returns)

        betas = []
        residuals = []
        resvars = []
        symbols = asset_returns.columns.tolist()
        factor_names = factor_returns.columns.tolist()
        X = add_constant(factor_returns.values)
        for symbol in symbols:
            y = asset_returns[symbol].values
            mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            if mask.sum() < self.min_obs:
                warnings.warn(f"Insufficient observations for {symbol}: {mask.sum()} < {self.min_obs}")
                betas.append(np.full(X.shape[1]-1, np.nan))
                residuals.append(np.full(asset_returns.shape[0], np.nan))
                resvars.append(np.nan)
                continue
            if self.regression_type == 'robust':
                model = RLM(y[mask], X[mask])
            else:
                model = OLS(y[mask], X[mask])
            results = model.fit()
            betas.append(results.params[1:])  # exclude intercept
            # Fill residuals with NaN where mask is False
            res_full = np.full(asset_returns.shape[0], np.nan)
            res_full[mask] = results.resid
            residuals.append(res_full)
            # Estimate residual variance for this asset
            resvars.append(np.nanvar(results.resid, ddof=1))
        beta_matrix = np.vstack(betas)

        factor_covar = cov_nearest(pd.DataFrame(factor_returns, columns=factor_names).cov().values)
        resvar = cov_nearest(pd.DataFrame(np.vstack(residuals).T).cov().values)

        residuals_matrix = np.vstack(residuals).T  # shape: (n_obs, n_assets)

        return LinearRiskModel(
            beta=beta_matrix,
            factor_covar=factor_covar,
            resvar=resvar,
            frequency=freq,
            symbols=symbols,
            factor_names=factor_names,
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            residual_returns=pd.DataFrame(residuals_matrix, index=asset_returns.index, columns=symbols)
        )