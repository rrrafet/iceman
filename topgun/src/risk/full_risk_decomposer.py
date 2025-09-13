import numpy as np
import pandas as pd
from typing import Union, Optional

from topgun.risk.model import RiskModel
from topgun.mappers import frequency_to_multiplier


class FullRiskDecomposer:
    """FullRiskDecomposer
    =====================
    *Allocation ⇢ Selection* • *Factor ⇢ Specific* decomposition with
    marginal contributions (MCAR) computed via the Euler principle.

    Extensions in this revision
    ---------------------------
    1. **Marginal contributions (MCAR)** and **Euler risk contributions** for
       every quadrant of the 2×2 grid:
       * allocation‑factor / allocation‑specific
       * selection‑factor / selection‑specific
    2. Asset‑level MCAR vectors (`mcar_*_asset`) and their Euler
       contributions (`ctr_*_asset`) which sum exactly to the corresponding
       component **volatility**.
    3. The legacy variance‑space and xσρ views are preserved.
    """

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(x: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = x.to_numpy()
        return np.squeeze(np.asarray(x, dtype=float))

    @staticmethod
    def _to_diag(mat_or_vec: np.ndarray) -> np.ndarray:
        return np.diag(mat_or_vec) if mat_or_vec.ndim == 1 else mat_or_vec

    @staticmethod
    def _safe_div(num: np.ndarray, denom: float):
        return np.zeros_like(num) if denom == 0 else num / denom

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
        # ====  Model pieces  =================================================
        self._model = model
        self.beta = self._to_numpy(model.beta)                 # N×K
        self.factor_covar = self._to_numpy(model.factor_covar) # K×K
        self.resvar = self._to_numpy(model.resvar)             # N or N×N
        self.frequency = model.frequency

        # Frequency multiplier for annualisation
        self._ann = 1.0
        if annualize:
            self._ann = frequency_to_multiplier.get(self.frequency.upper(), 1.0)

        # Factor vols (annualised)
        self._sigma_f = np.sqrt(np.diag(self.factor_covar) * self._ann)

        # ====  Weights  ======================================================
        self.w_p = self._to_numpy(portfolio_weights)
        self.w_b = self._to_numpy(benchmark_weights)
        self.w_d = self.w_p - self.w_b  # active weights

        # Store dimension helpers
        self.N = self.w_p.size
        self.K = self._sigma_f.size

        # Convenience matrices
        self._C_alloc_factor = self.beta @ self.factor_covar @ self.beta.T          # N×N
        self._D_alloc_specific = self._to_diag(self.resvar)                         # N×N

        # ====  Allocation – factor  =========================================
        # Variance and vol
        self._var_alloc_factor = float(self.w_d.T @ self._C_alloc_factor @ self.w_d * self._ann)
        self._vol_alloc_factor = float(np.sqrt(max(self._var_alloc_factor, 0.0)))

        # Marginal contribution (MCAR) vector & Euler contribution
        self._mcar_alloc_factor_asset = self._safe_div(
            self._C_alloc_factor @ self.w_d * self._ann, self._vol_alloc_factor
        )
        self._ctr_alloc_factor_asset = self.w_d * self._mcar_alloc_factor_asset  # sums to vol

        # ====  Allocation – specific  =======================================
        self._var_alloc_specific = float(self.w_d.T @ self._D_alloc_specific @ self.w_d * self._ann)
        self._vol_alloc_specific = float(np.sqrt(max(self._var_alloc_specific, 0.0)))

        self._mcar_alloc_specific_asset = self._safe_div(
            (self._D_alloc_specific @ self.w_d) * self._ann, self._vol_alloc_specific
        )
        self._ctr_alloc_specific_asset = self.w_d * self._mcar_alloc_specific_asset

        # ====  Selection – factor  ==========================================
        # ΔB matrix handling
        if selection_delta_beta_matrix is not None:
            self._delta_B_sel = np.asarray(selection_delta_beta_matrix, float)  # N×K
            self.gamma = self._delta_B_sel.T @ self.w_p                        # K,
        else:
            self._delta_B_sel = None
            self.gamma = (
                np.zeros(self.K) if selection_delta_beta is None else self._to_numpy(selection_delta_beta)
            )

        self._C_sel_factor = None
        if self._delta_B_sel is not None and self.gamma.any():
            self._C_sel_factor = self._delta_B_sel @ self.factor_covar @ self._delta_B_sel.T  # N×N
            self._var_sel_factor = float(self.w_p.T @ self._C_sel_factor @ self.w_p * self._ann)
            self._vol_sel_factor = np.sqrt(max(self._var_sel_factor, 0.0))
            self._mcar_sel_factor_asset = self._safe_div(
                (self._C_sel_factor @ self.w_p) * self._ann, self._vol_sel_factor
            )
            self._ctr_sel_factor_asset = self.w_p * self._mcar_sel_factor_asset
        else:
            self._var_sel_factor = 0.0
            self._vol_sel_factor = 0.0
            self._mcar_sel_factor_asset = np.zeros(self.N)
            self._ctr_sel_factor_asset = np.zeros(self.N)

        # ====  Selection – specific  ========================================
        sel_resvar_mat = self._to_diag(
            np.zeros_like(self.resvar) if selection_resvar is None else np.asarray(selection_resvar, float)
        )
        self._D_sel_specific = sel_resvar_mat
        self._var_sel_specific = float(self.w_p.T @ sel_resvar_mat @ self.w_p * self._ann)
        self._vol_sel_specific = np.sqrt(max(self._var_sel_specific, 0.0))
        self._mcar_sel_specific_asset = self._safe_div(
            (sel_resvar_mat @ self.w_p) * self._ann, self._vol_sel_specific
        )
        self._ctr_sel_specific_asset = self.w_p * self._mcar_sel_specific_asset

        # ====  Totals  =======================================================
        self._var_alloc = self._var_alloc_factor + self._var_alloc_specific
        self._var_sel = self._var_sel_factor + self._var_sel_specific
        self._var_total = self._var_alloc + self._var_sel
        self._vol_total = np.sqrt(max(self._var_total, 0.0))

    # ------------------------------------------------------------------
    # Expose component vols (annualised)
    # ------------------------------------------------------------------
    @property
    def vol_alloc_factor(self): return self._vol_alloc_factor
    @property
    def vol_alloc_specific(self): return self._vol_alloc_specific
    @property
    def vol_sel_factor(self): return self._vol_sel_factor
    @property
    def vol_sel_specific(self): return self._vol_sel_specific
    @property
    def vol_alloc(self): return self._vol_alloc_factor + self._vol_alloc_specific
    @property
    def vol_sel(self): return self._vol_sel_factor + self._vol_sel_specific
    @property
    def vol_total(self): return self._vol_total

    # ------------------------------------------------------------------
    # MCAR & Euler contributions – asset level
    # ------------------------------------------------------------------
    @property
    def mcar_alloc_factor_asset(self): return self._mcar_alloc_factor_asset
    @property
    def ctr_alloc_factor_asset(self): return self._ctr_alloc_factor_asset

    @property
    def mcar_alloc_specific_asset(self): return self._mcar_alloc_specific_asset
    @property
    def ctr_alloc_specific_asset(self): return self._ctr_alloc_specific_asset

    @property
    def mcar_sel_factor_asset(self): return self._mcar_sel_factor_asset
    @property
    def ctr_sel_factor_asset(self): return self._ctr_sel_factor_asset

    @property
    def mcar_sel_specific_asset(self): return self._mcar_sel_specific_asset
    @property
    def ctr_sel_specific_asset(self): return self._ctr_sel_specific_asset

    # ------------------------------------------------------------------
    # Backwards compatibility: total vol + model
    # ------------------------------------------------------------------
    @property
    def vol(self):
        return self._vol_total

    @property
    def model(self):
        return self._model


    def __repr__(self):
        # print a summary of the decomposer for all methods and properties, including private ones
        return (
            f"FullRiskDecomposer(vol_total={self.vol_total:.4f}, "
            f"vol_alloc_factor={self.vol_alloc_factor:.4f}, "
            f"vol_alloc_specific={self.vol_alloc_specific:.4f}, "
            f"vol_sel_factor={self.vol_sel_factor:.4f}, "
            f"vol_sel_specific={self.vol_sel_specific:.4f})"
        )