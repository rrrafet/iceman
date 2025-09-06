
import logging
from typing import Optional, List
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools import add_constant
from risk.model import RiskModel


logger = logging.getLogger(__name__)


class LinearRiskModel(RiskModel):
    """Concrete implementation of RiskModel for linear factor models."""
    
    def __init__(
        self,
        beta: np.ndarray,
        factor_covar: np.ndarray,
        resvar: np.ndarray,
        frequency: str,
        symbols: List[str],
        factor_names: List[str],
        asset_returns: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        residual_returns: Optional[pd.DataFrame] = None
    ):
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
    def beta(self) -> np.ndarray:
        """Factor loadings matrix"""
        return self._beta

    @property
    def factor_covar(self) -> np.ndarray:
        """Factor covariance matrix"""
        return self._factor_covar

    @property
    def resvar(self) -> np.ndarray:
        """Idiosyncratic (residual) variance matrix"""
        return self._resvar

    @property
    def covar(self) -> np.ndarray:
        """Total covariance matrix"""
        return self.beta @ self.factor_covar @ self.beta.T + self.resvar

    @property
    def frequency(self) -> str:
        """Data frequency (e.g., 'W-MON', 'D', etc.)"""
        return self._frequency

    @property
    def symbols(self) -> List[str]:
        """Asset symbols (asset_returns column names)"""
        return self._symbols

    @property
    def factor_names(self) -> List[str]:
        """Factor names (factor_returns column names)"""
        return self._factor_names

    @property
    def asset_returns(self) -> pd.DataFrame:
        """Original asset returns DataFrame"""
        if self._asset_returns is None:
            raise ValueError("Asset returns not available")
        return self._asset_returns

    @property
    def factor_returns(self) -> pd.DataFrame:
        """Original factor returns DataFrame"""
        if self._factor_returns is None:
            raise ValueError("Factor returns not available")
        return self._factor_returns

    @property
    def residual_returns(self) -> pd.DataFrame:
        """Residual returns DataFrame"""
        if self._residual_returns is None:
            raise ValueError("Residual returns not available")
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

    def _check_frequency(self, df1: pd.DataFrame, df2: pd.DataFrame) -> str:
        if self.freq is not None:
            return self.freq

        freq1 = pd.infer_freq(df1.index)
        freq2 = pd.infer_freq(df2.index)
        if freq1 != freq2:
            raise ValueError(f"Frequency mismatch: asset_returns={freq1}, factor_returns={freq2}")
        if freq1 is None:
            raise ValueError("Cannot infer frequency from asset_returns and factor_returns.")
        return freq1

    def _winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.winsorize_level is None:
            return df
        lower = df.quantile(self.winsorize_level / 2)
        upper = df.quantile(1 - self.winsorize_level / 2)
        return df.clip(lower=lower, upper=upper, axis=1)

    def _fit_regression(self, y: np.ndarray, X: np.ndarray):
        if self.regression_type == 'robust':
            return RLM(y, X).fit()
        return OLS(y, X).fit()

    def fit(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> LinearRiskModel:
        assert isinstance(asset_returns, pd.DataFrame)
        assert isinstance(factor_returns, pd.DataFrame)

        asset_returns, factor_returns = asset_returns.align(factor_returns, join='inner', axis=0)

        if not asset_returns.index.equals(factor_returns.index):
            raise ValueError("Asset and factor returns must be aligned on index.")

        if not self.freq:
            freq = self._check_frequency(asset_returns, factor_returns)
            if freq is None:
                raise ValueError("Cannot infer frequency from asset_returns index.")
        else:
            freq = self.freq
        
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
                logger.warning(f"Insufficient observations for {symbol}: {mask.sum()} < {self.min_obs}")
                betas.append(np.full(X.shape[1] - 1, np.nan))
                residuals.append(np.full(asset_returns.shape[0], np.nan))
                resvars.append(np.nan)
                continue

            results = self._fit_regression(y[mask], X[mask])
            betas.append(results.params[1:])  # Exclude intercept assuming factors are mean-centered

            res_full = np.full(asset_returns.shape[0], np.nan)
            res_full[mask] = results.resid
            residuals.append(res_full)
            resvars.append(np.nanvar(results.resid, ddof=1))

        beta_matrix = np.vstack(betas)
        factor_covar = cov_nearest(pd.DataFrame(factor_returns, columns=factor_names).cov().values)
        resvar = cov_nearest(pd.DataFrame(np.vstack(residuals).T).cov().values)
        residuals_matrix = np.vstack(residuals).T

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
