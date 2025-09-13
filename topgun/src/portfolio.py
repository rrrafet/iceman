from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any

class BasePortfolio(ABC):
    """
    Abstract base class for a portfolio.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Portfolio name"""
        pass

    @property
    @abstractmethod
    def date(self) -> Any:
        """Portfolio date (could be datetime or string)"""
        pass

    @property
    @abstractmethod
    def positions(self) -> pd.DataFrame:
        """Positions DataFrame (assets x position size)"""
        pass

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """List of asset symbols in the portfolio"""
        pass

    @property
    @abstractmethod
    def asset_names(self) -> list[str]:
        """List of asset names in the portfolio"""
        pass

    @property
    @abstractmethod
    def portfolio_weights(self) -> np.ndarray:
        """Portfolio weights as a numpy array"""
        pass

    @property
    @abstractmethod
    def benchmark_weights(self) -> np.ndarray:
        """Benchmark weights as a numpy array"""
        pass

    @property
    def active_weight(self) -> np.ndarray:
        """Active weights (portfolio_weights - benchmark_weights)"""
        return self.portfolio_weights - self.benchmark_weights

    @property
    @abstractmethod
    def asset_returns(self) -> pd.DataFrame:
        """Asset returns DataFrame (index: date, columns: symbols)"""
        pass

    @property
    def portfolio_returns(self) -> pd.Series:
        """Portfolio returns (asset returns weighted by portfolio_weights)"""
        # Assumes asset_returns columns are in the same order as symbols/weights
        return self.asset_returns @ self.portfolio_weights