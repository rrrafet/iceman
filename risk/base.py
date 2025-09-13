"""
Abstract base class for risk decomposition implementations.

This module provides the unified interface for portfolio risk decomposition,
harmonizing the different approaches used in RiskDecomposer and ActiveRiskDecomposer.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional


class RiskDecomposerBase(ABC):
    """
    Abstract base class for portfolio risk decomposition.
    
    This class provides a unified interface for different risk decomposition approaches,
    including traditional single-portfolio decomposition and active risk decomposition
    using Brinson-style allocation/selection analysis.
    
    All concrete implementations must provide methods for:
    - Risk calculation and contribution analysis
    - Validation of decomposition results  
    - Summary reporting of risk components
    
    The interface uses expressive property names that clearly indicate what each
    metric represents, improving code readability and reducing ambiguity.
    """
    
    # =========================================================================
    # CORE ABSTRACT PROPERTIES
    # =========================================================================
    
    @property
    @abstractmethod
    def portfolio_volatility(self) -> float:
        """
        Total portfolio volatility (risk).
        
        For single portfolio analysis, this is the absolute portfolio risk.
        For active risk analysis, this typically represents the total active risk.
        
        Returns
        -------
        float
            Annualized volatility/risk measure
        """
        pass
    
    @property
    @abstractmethod
    def portfolio_weights(self) -> np.ndarray:
        """
        Portfolio weights as a flat array.
        
        Returns
        -------
        np.ndarray
            Array of portfolio weights (N assets)
        """
        pass
    
    @property
    @abstractmethod
    def portfolio_factor_exposure(self) -> np.ndarray:
        """
        Portfolio's total factor exposures.
        
        Calculated as: B^T @ w where B is factor loadings matrix and w is weights.
        
        Returns
        -------
        np.ndarray
            Array of factor exposures (K factors)
        """
        pass
    
    # =========================================================================
    # RISK CONTRIBUTION ABSTRACT PROPERTIES
    # =========================================================================
    
    @property
    @abstractmethod
    def factor_risk_contribution(self) -> float:
        """
        Total contribution of factor risk to portfolio volatility.
        
        This represents the systematic risk component attributable to
        factor exposures and factor covariance.
        
        Returns
        -------
        float
            Factor risk contribution (volatility units)
        """
        pass
    
    @property
    @abstractmethod
    def specific_risk_contribution(self) -> float:
        """
        Total contribution of specific (idiosyncratic) risk to portfolio volatility.
        
        This represents the non-systematic risk component attributable to
        asset-specific factors not captured by the factor model.
        
        Returns
        -------
        float
            Specific risk contribution (volatility units)
        """
        pass
    
    @property
    @abstractmethod
    def asset_total_contributions(self) -> np.ndarray:
        """
        Total risk contribution by each asset to portfolio volatility.
        
        Each element represents how much each asset contributes to the
        overall portfolio risk, incorporating both factor and specific components.
        
        Returns
        -------
        np.ndarray
            Array of asset contributions (N assets, volatility units)
        """
        pass
    
    @property
    @abstractmethod
    def factor_contributions(self) -> np.ndarray:
        """
        Risk contribution by each factor to portfolio volatility.
        
        Each element represents how much each factor contributes to the
        overall portfolio risk through factor exposures and covariances.
        
        Returns
        -------
        np.ndarray
            Array of factor contributions (K factors, volatility units)
        """
        pass
    
    @property
    @abstractmethod
    def marginal_factor_contributions(self) -> np.ndarray:
        """
        Marginal contribution to risk from each factor.
        
        Represents the sensitivity of portfolio risk to small changes
        in factor exposures. Useful for risk attribution and optimization.
        
        Returns
        -------
        np.ndarray
            Array of marginal factor contributions (K factors, volatility units per unit exposure)
        """
        pass
    
    # =========================================================================
    # VALIDATION ABSTRACT METHODS
    # =========================================================================
    
    @abstractmethod
    def validate_contributions(self, tolerance: float = 1e-6) -> Dict:
        """
        Validate that risk decomposition results are mathematically consistent.
        
        Checks that asset contributions sum to total risk, factor and specific
        components are properly balanced, and other consistency conditions.
        
        Parameters
        ----------
        tolerance : float, default 1e-6
            Numerical tolerance for validation checks
            
        Returns
        -------
        Dict
            Dictionary containing validation results with 'passes' boolean
            and detailed information about any discrepancies
        """
        pass
    
    @abstractmethod
    def get_validation_summary(self, tolerance: float = 1e-6) -> str:
        """
        Generate a formatted summary of validation results.
        
        Parameters
        ----------
        tolerance : float, default 1e-6
            Numerical tolerance for validation checks
            
        Returns
        -------
        str
            Formatted validation summary report
        """
        pass
    
    @abstractmethod
    def risk_decomposition_summary(self) -> Dict:
        """
        Generate a comprehensive summary of risk decomposition results.
        
        Returns key risk metrics, contributions, and percentages in a
        structured format suitable for reporting and analysis.
        
        Returns
        -------
        Dict
            Dictionary containing comprehensive risk decomposition summary
        """
        pass
    
    # =========================================================================
    # OPTIONAL PROPERTIES (Default to None/NotImplemented for single-portfolio)
    # =========================================================================
    
    @property
    def benchmark_weights(self) -> Optional[np.ndarray]:
        """
        Benchmark weights (for active risk analysis only).
        
        Returns None for single-portfolio decomposition.
        
        Returns
        -------
        Optional[np.ndarray]
            Array of benchmark weights (N assets), or None if not applicable
        """
        return None
    
    @property
    def active_weights(self) -> Optional[np.ndarray]:
        """
        Active weights (portfolio - benchmark, for active risk analysis only).
        
        Returns None for single-portfolio decomposition.
        
        Returns
        -------
        Optional[np.ndarray]
            Array of active weights (N assets), or None if not applicable
        """
        return None
    
    @property
    def benchmark_factor_exposure(self) -> Optional[np.ndarray]:
        """
        Benchmark factor exposures (for active risk analysis only).
        
        Returns None for single-portfolio decomposition.
        
        Returns
        -------
        Optional[np.ndarray]
            Array of benchmark factor exposures (K factors), or None if not applicable
        """
        return None
    
    @property
    def active_factor_exposure(self) -> Optional[np.ndarray]:
        """
        Active factor exposures (portfolio - benchmark, for active risk analysis only).
        
        Returns None for single-portfolio decomposition.
        
        Returns
        -------
        Optional[np.ndarray]
            Array of active factor exposures (K factors), or None if not applicable
        """
        return None
    
    # =========================================================================
    # ACTIVE RISK SPECIFIC PROPERTIES (Default to NotImplemented)
    # =========================================================================
    
    @property
    def total_active_risk(self) -> Optional[float]:
        """
        Total active risk (for active risk decomposition only).
        
        Returns None for single-portfolio decomposition.
        
        Returns
        -------
        Optional[float]
            Total active risk (volatility units), or None if not applicable
        """
        return None
    
    @property
    def allocation_factor_risk(self) -> Optional[float]:
        """
        Allocation factor risk component (for active risk decomposition only).
        
        Risk from tilting weights relative to benchmark while maintaining
        benchmark factor characteristics.
        
        Returns
        -------
        Optional[float]
            Allocation factor risk, or None if not applicable
        """
        return None
    
    @property
    def allocation_specific_risk(self) -> Optional[float]:
        """
        Allocation specific risk component (for active risk decomposition only).
        
        Specific risk from tilting weights relative to benchmark.
        
        Returns
        -------
        Optional[float]
            Allocation specific risk, or None if not applicable
        """
        return None
    
    @property
    def selection_factor_risk(self) -> Optional[float]:
        """
        Selection factor risk component (for active risk decomposition only).
        
        Risk from choosing different factor exposures within asset groups.
        
        Returns
        -------
        Optional[float]
            Selection factor risk, or None if not applicable
        """
        return None
    
    @property
    def selection_specific_risk(self) -> Optional[float]:
        """
        Selection specific risk component (for active risk decomposition only).
        
        Specific risk from choosing different securities within groups.
        
        Returns
        -------
        Optional[float]
            Selection specific risk, or None if not applicable
        """
        return None
    
    @property
    def total_allocation_risk(self) -> Optional[float]:
        """
        Total allocation risk (allocation factor + specific, for active risk decomposition only).
        
        Returns
        -------
        Optional[float]
            Total allocation risk, or None if not applicable
        """
        return None
    
    @property
    def total_selection_risk(self) -> Optional[float]:
        """
        Total selection risk (selection factor + specific, for active risk decomposition only).
        
        Returns
        -------
        Optional[float]
            Total selection risk, or None if not applicable
        """
        return None
    
    # =========================================================================
    # ASSET-LEVEL ACTIVE CONTRIBUTIONS (Default to None)
    # =========================================================================
    
    @property
    def asset_allocation_factor_contributions(self) -> Optional[np.ndarray]:
        """
        Asset-level contributions to allocation factor risk (for active risk decomposition only).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of asset allocation factor contributions, or None if not applicable
        """
        return None
    
    @property
    def asset_allocation_specific_contributions(self) -> Optional[np.ndarray]:
        """
        Asset-level contributions to allocation specific risk (for active risk decomposition only).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of asset allocation specific contributions, or None if not applicable
        """
        return None
    
    @property
    def asset_selection_factor_contributions(self) -> Optional[np.ndarray]:
        """
        Asset-level contributions to selection factor risk (for active risk decomposition only).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of asset selection factor contributions, or None if not applicable
        """
        return None
    
    @property
    def asset_selection_specific_contributions(self) -> Optional[np.ndarray]:
        """
        Asset-level contributions to selection specific risk (for active risk decomposition only).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of asset selection specific contributions, or None if not applicable
        """
        return None
    
    # =========================================================================
    # FACTOR-LEVEL ACTIVE CONTRIBUTIONS (Default to None)
    # =========================================================================
    
    @property
    def factor_allocation_contributions(self) -> Optional[np.ndarray]:
        """
        Factor-level contributions to allocation risk (for active risk decomposition only).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of factor allocation contributions, or None if not applicable
        """
        return None
    
    @property
    def factor_selection_contributions(self) -> Optional[np.ndarray]:
        """
        Factor-level contributions to selection risk (for active risk decomposition only).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of factor selection contributions, or None if not applicable
        """
        return None
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def print_validation_summary(self, tolerance: float = 1e-6) -> None:
        """
        Print validation summary to console.
        
        Parameters
        ----------
        tolerance : float, default 1e-6
            Numerical tolerance for validation checks
        """
        summary = self.get_validation_summary(tolerance)
        print(summary)
    
    def is_active_decomposer(self) -> bool:
        """
        Check if this is an active risk decomposer.
        
        Returns
        -------
        bool
            True if this decomposer supports active risk analysis
        """
        return self.total_active_risk is not None
    
    def get_risk_summary_table(self) -> pd.DataFrame:
        """
        Generate a formatted DataFrame summarizing key risk metrics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with risk component names and values
        """
        
        # Create base summary for all decomposers
        data = {
            'Risk Component': ['Portfolio Volatility', 'Factor Risk', 'Specific Risk'],
            'Value': [
                self.portfolio_volatility,
                self.factor_risk_contribution,
                self.specific_risk_contribution
            ],
            'Percentage': [
                100.0,
                100.0 * self.factor_risk_contribution / self.portfolio_volatility,
                100.0 * self.specific_risk_contribution / self.portfolio_volatility
            ]
        }
        
        # Add active risk components if available
        if self.is_active_decomposer():
            active_data = {
                'Risk Component': [
                    'Total Active Risk',
                    'Allocation Factor Risk', 
                    'Allocation Specific Risk',
                    'Selection Factor Risk',
                    'Selection Specific Risk'
                ],
                'Value': [
                    self.total_active_risk,
                    self.allocation_factor_risk,
                    self.allocation_specific_risk,
                    self.selection_factor_risk,
                    self.selection_specific_risk
                ],
                'Percentage': [
                    100.0,
                    100.0 * (self.allocation_factor_risk or 0) / (self.total_active_risk or 1),
                    100.0 * (self.allocation_specific_risk or 0) / (self.total_active_risk or 1),
                    100.0 * (self.selection_factor_risk or 0) / (self.total_active_risk or 1),
                    100.0 * (self.selection_specific_risk or 0) / (self.total_active_risk or 1)
                ]
            }
            
            # Combine data
            data['Risk Component'].extend(active_data['Risk Component'])
            data['Value'].extend(active_data['Value'])
            data['Percentage'].extend(active_data['Percentage'])
        
        return pd.DataFrame(data)