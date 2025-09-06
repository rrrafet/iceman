from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any

class RiskModel(ABC):
    @property
    @abstractmethod
    def beta(self) -> np.ndarray:
        """Factor loadings matrix"""
        pass

    @property
    @abstractmethod
    def factor_covar(self) -> np.ndarray:
        """Factor covariance matrix"""
        pass

    @property
    @abstractmethod
    def resvar(self) -> np.ndarray:
        """Idiosyncratic (residual) variance matrix"""
        pass

    @property
    @abstractmethod
    def covar(self) -> np.ndarray:
        """Total covariance matrix"""
        pass

    @property
    @abstractmethod
    def frequency(self) -> str:
        """Data frequency (e.g., 'W-MON', 'D', etc.)"""
        pass

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """Asset symbols (asset_returns column names)"""
        pass

    @property
    @abstractmethod
    def factor_names(self) -> list[str]:
        """Factor names (factor_returns column names)"""
        pass

    @property
    @abstractmethod
    def asset_returns(self) -> pd.DataFrame:
        """Original asset returns DataFrame"""
        pass

    @property
    @abstractmethod
    def factor_returns(self) -> pd.DataFrame:
        """Original factor returns DataFrame"""
        pass

    @property
    @abstractmethod
    def residual_returns(self) -> pd.DataFrame:
        """Residual returns DataFrame or array"""
        pass