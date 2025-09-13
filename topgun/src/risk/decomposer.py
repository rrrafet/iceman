import numpy as np
import pandas as pd
from typing import Union
from topgun.risk.model import RiskModel
from topgun.mappers import frequency_to_multiplier


class RiskDecomposer:
    """
    RiskDecomposer provides methods to decompose portfolio risk into asset-level and factor-level contributions.

    This class uses a risk model (typically a factor model) and portfolio weights to compute various risk decomposition metrics, including marginal, percent, and absolute contributions to risk from both assets and factors. It supports annualization of risk metrics based on the frequency of returns.

    Attributes:
        model (RiskModel): The risk model containing factor exposures, covariances, and residual variances.
        beta (np.ndarray): Factor exposure matrix from the risk model.
        factor_covar (np.ndarray): Factor covariance matrix from the risk model.
        resvar (np.ndarray): Residual (idiosyncratic) variance matrix from the risk model.
        covar (np.ndarray): Full asset covariance matrix from the risk model.
        frequency (str): Frequency of returns (e.g., 'DAILY', 'MONTHLY').
        annualization_multiplier (float): Multiplier to annualize risk metrics.
        w (np.ndarray): Portfolio weights.

    Properties:
        vol: Portfolio volatility (possibly annualized).
        mctr: Marginal contribution to risk per asset.
        pctr: Percent contribution to risk per asset.
        ctr: Absolute contribution to risk per asset.
        ctr_factor: Contribution to risk from factor risk per asset.
        ctr_idio: Contribution to risk from idiosyncratic risk per asset.
        factor_exposure: Portfolio-level exposure to each factor.
        mfctr: Marginal contribution to risk per factor.
        pfctr: Percent contribution to risk per factor.
        fctr: Absolute contribution to risk per factor.
        ctr_asset_factor: Contribution to risk for each asset per factor.

    Args:
        model (RiskModel): The risk model object containing necessary risk parameters.
        weights (np.ndarray or pd.Series or pd.DataFrame): Portfolio weights.
        annualize (bool, optional): Whether to annualize risk metrics. Defaults to True.

    Raises:
        ValueError: If input dimensions are inconsistent or required model attributes are missing.

    Example:
        decomposer = RiskDecomposer(model, weights)
        volatility = decomposer.vol
        asset_contributions = decomposer.ctr
        factor_contributions = decomposer.cfctr
    
    """
    def __init__(
        self,
        model: RiskModel,
        weights: Union[np.ndarray, pd.Series, pd.DataFrame],
        annualize: bool = True
    ) -> None:
        self._model = model
        self.beta = model.beta
        self.factor_covar = model.factor_covar
        self.resvar = model.resvar
        self.covar = model.covar
        self.frequency = model.frequency
        self.annualization_multiplier = 1.0

        if annualize:
            self.annualization_multiplier = frequency_to_multiplier.get(self.frequency.upper(), 1.0)

        if isinstance(weights, (pd.Series, pd.DataFrame)):
            weights = weights.to_numpy()
        self.w = weights

    @property
    def model(self) -> RiskModel:
        return self._model
    

    @property
    def vol(self) -> float:
        """
        Calculate the estimated volatility using allocation weights.
        :return: ex-ante volatility (not annualized).
        """
        return np.sqrt(self.annualization_multiplier) * np.sqrt(self.w.T @ self.covar @ self.w)

    @property
    def mctr(self) -> np.ndarray:
        """
        Calculate marginal contribution to risk per asset.
        :return: array with marginal contribution per asset.
        """
        return self.annualization_multiplier * (self.covar @ self.w) / self.vol

    @property
    def pctr(self) -> np.ndarray:
        """
        Calculate percent contribution to risk per asset.
        :return: array with percent contribution per asset (should sum to 1).
        """
        return (self.w * self.mctr) / self.vol

    @property
    def ctr(self) -> np.ndarray:
        """
        Calculate contribution to volatility per asset.
        :return: array with contribution to volatility per asset.
        """
        return self.w * self.mctr

    @property
    def ctr_factor(self) -> np.ndarray:
        """
        Calculate the contribution to volatility from factor risk.
        :return: array with contribution to factor risk per asset.
        """
        ff = self.beta @ self.factor_covar @ self.beta.T
        return self.annualization_multiplier * (self.w * (ff @ self.w) / self.vol)

    @property
    def ctr_idio(self) -> np.ndarray:
        """
        Calculate the contribution to volatility from idiosyncratic risk.
        :return: array with contribution to idiosyncratic risk per asset.
        """
        return self.annualization_multiplier * (self.w * (self.resvar @ self.w) / self.vol)

    @property
    def factor_exposure(self) -> np.ndarray:
        """
        Calculate portfolio level beta exposure for each factor.
        :return: array of portfolio beta to each factor.
        """
        return self.beta.T @ self.w

    @property
    def mfctr(self) -> np.ndarray:
        """
        Calculate marginal contribution to volatility per factor.
        :return: array of marginal contribution to volatility per factor.
        """
        return self.annualization_multiplier * (self.factor_covar.dot(self.factor_exposure)) / self.vol

    @property
    def pfctr(self) -> np.ndarray:
        """
        Calculate percent contribution to volatility per factor.
        :return: array of percent contribution to volatility (should sum to percent factor risk).
        """
        return (self.factor_exposure * self.mfctr) / self.vol

    @property
    def fctr(self) -> np.ndarray:
        """
        Calculate contribution to volatility per factor.
        :return: array of contribution to volatility (should sum to portfolio factor volatility).
        """
        return self.factor_exposure * self.mfctr

    @property
    def ctr_asset_factor(self) -> np.ndarray:
        """
        Calculate contribution to volatility for assets per each factor.
        :return: contribution by asset and factor.
        """
        first = np.diag(self.factor_exposure)
        second = self.factor_covar @ self.beta.T / self.vol
        third = np.diag(self.w)
        return self.annualization_multiplier * (first @ second @ third).T


import numpy as np
import pandas as pd
from typing import Union, Optional

from topgun.risk.model import RiskModel
from topgun.mappers import frequency_to_multiplier


class FullRiskDecomposer:
    """FullRiskDecomposer — *complete* allocation ⇢ selection, factor ⇢ specific breakdown

    This version exposes **full crossed contribution matrices** so you can see
    *how every asset interacts with every factor* in both the **allocation**
    and **selection** parts of active risk.

    New public attributes
    ---------------------
    ▸ ``alloc_factor_contrib_matrix``  → ``(N×K)`` allocation‑factor contribs
    ▸ ``sel_factor_contrib_matrix``    → ``(N×K)`` selection‑factor contribs
       (returns ``None`` if ``selection_delta_beta_matrix`` not supplied)
    ▸ corresponding ``*_pct_matrix`` helpers (each element ÷ component total)

    All matrices are in **annualised variance units** and sum to the relevant
    component variance.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(x: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = x.to_numpy()
        return np.squeeze(np.asarray(x, dtype=float))

    @staticmethod
    def _to_diag(mat_or_vec: np.ndarray) -> np.ndarray:
        return np.diag(mat_or_vec) if mat_or_vec.ndim == 1 else mat_or_vec

    # ------------------------------------------------------------------
    # Initialiser
    # ------------------------------------------------------------------
    def __init__(
        self,
        model: RiskModel,
        portfolio_weights: Union[np.ndarray, pd.Series, pd.DataFrame],
        benchmark_weights: Union[np.ndarray, pd.Series, pd.DataFrame],
        *,
        selection_delta_beta: Optional[np.ndarray] = None,
        selection_resvar: Optional[np.ndarray] = None,
        selection_delta_beta_matrix: Optional[np.ndarray] = None,
        annualize: bool = True,
    ) -> None:
        # Risk‑model bits
        self._model: RiskModel = model
        self.beta: np.ndarray = self._to_numpy(model.beta)              # N×K
        self.factor_covar: np.ndarray = self._to_numpy(model.factor_covar)  # K×K
        self.resvar: np.ndarray = self._to_numpy(model.resvar)
        self.frequency: str = model.frequency

        # Annualisation multiplier
        self._ann_mult: float = 1.0
        if annualize:
            self._ann_mult = frequency_to_multiplier.get(self.frequency.upper(), 1.0)

        # Weights
        self.w_p: np.ndarray = self._to_numpy(portfolio_weights)
        self.w_b: np.ndarray = self._to_numpy(benchmark_weights)
        self.w_d: np.ndarray = self.w_p - self.w_b

        # ------------------------------------------------------------------
        # Allocation component
        # ------------------------------------------------------------------
        self._delta_beta_alloc: np.ndarray = self.beta.T @ self.w_d  # K,
        _ΩΔβ = self.factor_covar @ self._delta_beta_alloc            # K,
        # Scalar variance
        self._var_alloc_factor_raw: float = float(self._delta_beta_alloc @ _ΩΔβ)
        # Contribution matrix N×K  : w_d_i β_ik (ΩΔβ)_k
        self._alloc_factor_contrib_matrix_raw: np.ndarray = (
            (self.w_d[:, None] * self.beta) * _ΩΔβ[None, :]
        )
        # Asset & factor marginalisations
        self._alloc_factor_contrib_asset_raw: np.ndarray = self._alloc_factor_contrib_matrix_raw.sum(axis=1)
        self._alloc_factor_contrib_factor_raw: np.ndarray = self._alloc_factor_contrib_matrix_raw.sum(axis=0)

        # Specific part (allocation)
        resvar_mat = self._to_diag(self.resvar)
        self._var_alloc_specific_raw: float = float(self.w_d.T @ resvar_mat @ self.w_d)
        self._alloc_specific_contrib_asset_raw: np.ndarray = (self.w_d ** 2) * np.diag(resvar_mat)

        # ------------------------------------------------------------------
        # Selection component
        # ------------------------------------------------------------------
        self._delta_B_sel: Optional[np.ndarray] = None
        if selection_delta_beta_matrix is not None:
            self._delta_B_sel = np.asarray(selection_delta_beta_matrix, dtype=float)
            self.gamma: np.ndarray = self._delta_B_sel.T @ self.w_p
        else:
            self.gamma: np.ndarray = (
                np.zeros(self.factor_covar.shape[0]) if selection_delta_beta is None else self._to_numpy(selection_delta_beta)
            )

        _Ωγ = self.factor_covar @ self.gamma
        self._var_sel_factor_raw: float = float(self.gamma @ _Ωγ)
        self._sel_factor_contrib_factor_raw: np.ndarray = self.gamma * _Ωγ

        if self._delta_B_sel is not None:
            self._sel_factor_contrib_matrix_raw: np.ndarray = (
                (self.w_p[:, None] * self._delta_B_sel) * _Ωγ[None, :]
            )
            self._sel_factor_contrib_asset_raw: np.ndarray = self._sel_factor_contrib_matrix_raw.sum(axis=1)
        else:
            self._sel_factor_contrib_matrix_raw = None
            self._sel_factor_contrib_asset_raw = None

        # Specific (selection)
        sel_resvar_mat = self._to_diag(np.zeros_like(self.resvar) if selection_resvar is None else np.asarray(selection_resvar, float))
        self._var_sel_specific_raw: float = float(self.w_p.T @ sel_resvar_mat @ self.w_p)
        self._sel_specific_contrib_asset_raw: np.ndarray = (self.w_p ** 2) * np.diag(sel_resvar_mat)

        # ------------------------------------------------------------------
        # Annualise everything
        # ------------------------------------------------------------------
        def _ann(x): return x * self._ann_mult

        self._var_alloc_factor = _ann(self._var_alloc_factor_raw)
        self._var_alloc_specific = _ann(self._var_alloc_specific_raw)
        self._var_sel_factor = _ann(self._var_sel_factor_raw)
        self._var_sel_specific = _ann(self._var_sel_specific_raw)

        self._alloc_factor_contrib_matrix = _ann(self._alloc_factor_contrib_matrix_raw)
        self._alloc_factor_contrib_asset = _ann(self._alloc_factor_contrib_asset_raw)
        self._alloc_factor_contrib_factor = _ann(self._alloc_factor_contrib_factor_raw)
        self._alloc_specific_contrib_asset = _ann(self._alloc_specific_contrib_asset_raw)

        if self._sel_factor_contrib_matrix_raw is not None:
            self._sel_factor_contrib_matrix = _ann(self._sel_factor_contrib_matrix_raw)
            self._sel_factor_contrib_asset = _ann(self._sel_factor_contrib_asset_raw)
        else:
            self._sel_factor_contrib_matrix = None
            self._sel_factor_contrib_asset = None
        self._sel_factor_contrib_factor = _ann(self._sel_factor_contrib_factor_raw)
        self._sel_specific_contrib_asset = _ann(self._sel_specific_contrib_asset_raw)

        # Totals
        self._var_alloc = self._var_alloc_factor + self._var_alloc_specific
        self._var_sel = self._var_sel_factor + self._var_sel_specific
        self._var_total = self._var_alloc + self._var_sel

    # ------------------------------------------------------------------
    # Variance accessors
    # ------------------------------------------------------------------
    @property
    def var_alloc_factor(self): return self._var_alloc_factor
    @property
    def var_alloc_specific(self): return self._var_alloc_specific
    @property
    def var_alloc(self): return self._var_alloc
    @property
    def var_sel_factor(self): return self._var_sel_factor
    @property
    def var_sel_specific(self): return self._var_sel_specific
    @property
    def var_sel(self): return self._var_sel
    @property
    def var_total(self): return self._var_total

    # ------------------------------------------------------------------
    # Contribution matrices / vectors
    # ------------------------------------------------------------------
    # Allocation – factor (matrix, asset, factor)
    @property
    def alloc_factor_contrib_matrix(self) -> np.ndarray:
        """``(N, K)`` matrix – asset×factor contributions to allocation‑factor variance."""
        return self._alloc_factor_contrib_matrix

    @property
    def alloc_factor_contrib_by_asset(self) -> np.ndarray:
        return self._alloc_factor_contrib_asset

    @property
    def alloc_factor_contrib_by_factor(self) -> np.ndarray:
        return self._alloc_factor_contrib_factor

    # Selection – factor
    @property
    def sel_factor_contrib_matrix(self) -> Optional[np.ndarray]:
        """``(N, K)`` matrix – asset×factor contributions to selection‑factor variance.
        Returns ``None`` when the ΔB matrix was not provided."""
        return self._sel_factor_contrib_matrix

    @property
    def sel_factor_contrib_by_asset(self) -> Optional[np.ndarray]:
        return self._sel_factor_contrib_asset

    @property
    def sel_factor_contrib_by_factor(self) -> np.ndarray:
        return self._sel_factor_contrib_factor

    # Specific parts (unchanged)
    @property
    def alloc_specific_contrib_by_asset(self) -> np.ndarray:
        return self._alloc_specific_contrib_asset

    @property
    def sel_specific_contrib_by_asset(self) -> np.ndarray:
        return self._sel_specific_contrib_asset

    # ------------------------------------------------------------------
    # Percent‑of‑component helpers
    # ------------------------------------------------------------------
    def _pct(self, contrib: np.ndarray, total: float):
        return np.zeros_like(contrib) if total == 0 else contrib / total

    # Allocation – factor pct matrices/vectors
    @property
    def alloc_factor_pct_matrix(self) -> np.ndarray:
        return self._pct(self.alloc_factor_contrib_matrix, self.var_alloc_factor)

    @property
    def alloc_factor_pct_by_asset(self) -> np.ndarray:
        return self._pct(self.alloc_factor_contrib_by_asset, self.var_alloc_factor)

    @property
    def alloc_factor_pct_by_factor(self) -> np.ndarray:
        return self._pct(self.alloc_factor_contrib_by_factor, self.var_alloc_factor)

    # Selection – factor pct
    @property
    def sel_factor_pct_matrix(self) -> Optional[np.ndarray]:
        if self.sel_factor_contrib_matrix is None:
            return None
        return self._pct(self.sel_factor_contrib_matrix, self.var_sel_factor)

    @property
    def sel_factor_pct_by_asset(self) -> Optional[np.ndarray]:
        if self.sel_factor_contrib_by_asset is None:
            return None
        return self._pct(self.sel_factor_contrib_by_asset, self.var_sel_factor)

    @property
    def sel_factor_pct_by_factor(self) -> np.ndarray:
        return self._pct(self.sel_factor_contrib_by_factor, self.var_sel_factor)

    # ------------------------------------------------------------------
    # Vol alias for backwards compatibility
    # ------------------------------------------------------------------
    @property
    def vol(self):  # noqa: D401
        return float(np.sqrt(max(self.var_total, 0.0)))

    # Expose model
    @property
    def model(self):
        return self._model
