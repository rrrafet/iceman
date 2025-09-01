"""
Legacy compatibility layer for the simplified risk system.

This module provides backwards compatibility wrappers so existing code continues
to work while users migrate to the simplified API. The wrappers translate between
the old complex interfaces and the new simple functions.
"""

import warnings
import numpy as np
from typing import Dict, Any, Optional, Union, List

from .risk_analysis import analyze_portfolio_risk, analyze_active_risk, RiskResult
from .context import RiskContext, SingleModelContext, MultiModelContext
from .model import RiskModel


# Legacy type alias for backward compatibility
RiskAnalysis = RiskResult  # Old name -> new name


def create_single_model_context(
    model: RiskModel, 
    weights: Union[np.ndarray, Dict[str, float]],
    annualize: bool = True
) -> RiskResult:
    """
    Legacy function that creates a single model context and performs analysis.
    
    This maintains backward compatibility with the old context-based API.
    
    Parameters
    ----------
    model : RiskModel
        Risk model
    weights : Union[np.ndarray, Dict[str, float]]
        Portfolio weights
    annualize : bool, default True
        Whether to annualize results
        
    Returns
    -------
    RiskResult
        Analysis results (same as RiskAnalysis for backward compatibility)
    """
    warnings.warn(
        "create_single_model_context is deprecated. Use analyze_portfolio_risk() directly.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return analyze_portfolio_risk(model, weights, annualize=annualize)


def create_active_risk_context(
    portfolio_model: RiskModel,
    benchmark_model: RiskModel, 
    portfolio_weights: Union[np.ndarray, Dict[str, float]],
    benchmark_weights: Union[np.ndarray, Dict[str, float]],
    active_model: Optional[RiskModel] = None,
    cross_covar: Optional[np.ndarray] = None,
    annualize: bool = True
) -> RiskResult:
    """
    Legacy function that creates active risk context and performs analysis.
    
    This maintains backward compatibility with the old context-based API.
    
    Parameters
    ---------- 
    portfolio_model : RiskModel
        Portfolio risk model
    benchmark_model : RiskModel
        Benchmark risk model
    portfolio_weights : Union[np.ndarray, Dict[str, float]]
        Portfolio weights
    benchmark_weights : Union[np.ndarray, Dict[str, float]]
        Benchmark weights
    active_model : RiskModel, optional
        Active risk model
    cross_covar : np.ndarray, optional
        Cross-covariance matrix
    annualize : bool, default True
        Whether to annualize results
        
    Returns
    -------
    RiskResult
        Analysis results (same as RiskAnalysis for backward compatibility)
    """
    warnings.warn(
        "create_active_risk_context is deprecated. Use analyze_active_risk() directly.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return analyze_active_risk(
        portfolio_model, benchmark_model, 
        portfolio_weights, benchmark_weights,
        active_model=active_model,
        cross_covar=cross_covar,
        annualize=annualize
    )


class LegacyRiskDecomposer:
    """
    Backwards compatibility wrapper for the old RiskDecomposer interface.
    
    This class provides the same interface as the original RiskDecomposer,
    but internally uses the simplified risk analysis functions. This allows
    existing code to continue working without modification.
    """
    
    def __init__(self, context: RiskContext):
        """
        Initialize with legacy context.
        
        Parameters
        ----------
        context : RiskContext  
            Legacy risk context (SingleModelContext or MultiModelContext)
        """
        warnings.warn(
            "LegacyRiskDecomposer is deprecated. Use analyze_portfolio_risk() or analyze_active_risk() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self._context = context
        
        # Perform analysis based on context type
        if isinstance(context, SingleModelContext):
            self._result = analyze_portfolio_risk(
                context.model,
                context.weights,
                annualize=context.annualize
            )
            self._analysis_type = "portfolio"
        elif isinstance(context, MultiModelContext):
            self._result = analyze_active_risk(
                context.portfolio_model,
                context.benchmark_model, 
                context.portfolio_weights,
                context.benchmark_weights,
                active_model=getattr(context, 'active_model', None),
                cross_covar=getattr(context, 'cross_covar', None),
                annualize=context.annualize
            )
            self._analysis_type = "active"
        else:
            raise ValueError(f"Unsupported context type: {type(context)}")
    
    # Legacy property interfaces
    @property
    def portfolio_volatility(self) -> float:
        """Total portfolio/active risk volatility."""
        return self._result.total_risk
    
    @property
    def factor_risk_contribution(self) -> float:
        """Factor risk component."""
        return self._result.factor_risk_contribution
    
    @property
    def specific_risk_contribution(self) -> float:
        """Specific risk component."""
        return self._result.specific_risk_contribution
    
    @property
    def total_risk(self) -> float:
        """Alias for portfolio_volatility."""
        return self._result.total_risk
    
    @property
    def factor_contributions(self) -> Dict[str, float]:
        """Factor-level risk contributions."""
        return self._result.factor_contributions
    
    @property
    def asset_contributions(self) -> Dict[str, float]:
        """Asset-level risk contributions."""
        return self._result.asset_contributions
    
    @property
    def factor_exposures(self) -> Dict[str, float]:
        """Factor exposures."""
        return self._result.factor_exposures
    
    # Active risk specific properties
    @property
    def total_active_risk(self) -> float:
        """Total active risk (for active analysis)."""
        if self._analysis_type == "active":
            return self._result.total_risk
        return 0.0
    
    @property
    def allocation_factor_risk(self) -> Optional[float]:
        """Allocation factor risk component."""
        return self._result.allocation_factor_risk
    
    @property
    def allocation_specific_risk(self) -> Optional[float]:
        """Allocation specific risk component."""
        return self._result.allocation_specific_risk
    
    @property
    def selection_factor_risk(self) -> Optional[float]:
        """Selection factor risk component."""
        return self._result.selection_factor_risk
    
    @property
    def selection_specific_risk(self) -> Optional[float]:
        """Selection specific risk component."""
        return self._result.selection_specific_risk
    
    # Legacy method interfaces
    def risk_decomposition_summary(self) -> Dict[str, Any]:
        """
        Return risk decomposition summary in legacy format.
        
        Returns
        -------
        Dict[str, Any]
            Risk decomposition metrics in legacy format
        """
        summary = {
            'portfolio_volatility': self._result.total_risk,
            'factor_risk_contribution': self._result.factor_risk_contribution,
            'specific_risk_contribution': self._result.specific_risk_contribution,
            'factor_risk_percentage': self._result.factor_risk_pct,
            'specific_risk_percentage': self._result.specific_risk_pct,
            'analysis_type': self._result.analysis_type,
            'annualized': self._result.annualized
        }
        
        # Add active risk specific metrics
        if self._analysis_type == "active":
            summary.update({
                'total_active_risk': self._result.total_risk,
                'allocation_factor_risk': self._result.allocation_factor_risk,
                'allocation_specific_risk': self._result.allocation_specific_risk,
                'selection_factor_risk': self._result.selection_factor_risk,
                'selection_specific_risk': self._result.selection_specific_risk,
                'cross_correlation_risk_contribution': self._result.cross_correlation_risk_contribution,
                'cross_correlation_volatility': self._result.cross_correlation_volatility
            })
        
        return summary
    
    def get_factor_contributions(self) -> np.ndarray:
        """Return factor contributions as numpy array (legacy format)."""
        return np.array(list(self._result.factor_contributions.values()))
    
    def get_asset_contributions(self) -> np.ndarray:
        """Return asset contributions as numpy array (legacy format)."""
        return np.array(list(self._result.asset_contributions.values()))
    
    def get_factor_exposures(self) -> np.ndarray:
        """Return factor exposures as numpy array (legacy format)."""
        return np.array(list(self._result.factor_exposures.values()))
    
    def calculate_risk_decomposition(self) -> Dict[str, Any]:
        """Legacy method name - returns the same as risk_decomposition_summary()."""
        return self.risk_decomposition_summary()


class LegacyRiskResultSchema:
    """
    Backwards compatibility wrapper for the old RiskResultSchema interface.
    
    This provides access to RiskAnalysis data using the old schema-style interface,
    allowing existing UI and reporting code to work without modification.
    """
    
    def __init__(self, risk_analysis: RiskAnalysis):
        """
        Initialize from RiskAnalysis result.
        
        Parameters
        ----------
        risk_analysis : RiskAnalysis
            Simple risk analysis result
        """
        warnings.warn(
            "LegacyRiskResultSchema is deprecated. Access RiskAnalysis properties directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self._result = risk_analysis
        self._lens = risk_analysis.analysis_type
    
    def get_lens_core_metrics(self, lens: str) -> Dict[str, float]:
        """Get core metrics for a lens (legacy interface)."""
        return {
            'total_risk': self._result.total_risk,
            'factor_risk_contribution': self._result.factor_risk_contribution, 
            'specific_risk_contribution': self._result.specific_risk_contribution
        }
    
    def get_lens_factor_contributions(self, lens: str) -> Dict[str, float]:
        """Get factor contributions for a lens."""
        return self._result.factor_contributions
    
    def get_lens_asset_contributions(self, lens: str) -> Dict[str, float]:
        """Get asset contributions for a lens."""
        return self._result.asset_contributions
    
    def get_lens_factor_exposures(self, lens: str) -> Dict[str, float]:
        """Get factor exposures for a lens."""
        return self._result.factor_exposures
    
    def get_validation_results(self) -> Dict[str, Any]:
        """Get validation results (legacy format)."""
        return {
            'passes': self._result.validation_passed,
            'message': self._result.validation_message
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (legacy export format)."""
        return self._result.to_dict()
    
    # Property access for common metrics
    @property  
    def total_risk(self) -> float:
        return self._result.total_risk
    
    @property
    def factor_risk(self) -> float:
        return self._result.factor_risk_contribution
    
    @property
    def specific_risk(self) -> float:
        return self._result.specific_risk_contribution


def create_single_model_context(
    model: RiskModel,
    weights: Union[np.ndarray, Dict[str, float]],
    annualize: bool = True
) -> SingleModelContext:
    """
    Helper function to create SingleModelContext (for backwards compatibility).
    
    Note: It's better to use analyze_portfolio_risk() directly.
    """
    warnings.warn(
        "create_single_model_context is deprecated. Use analyze_portfolio_risk() directly.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if isinstance(weights, dict):
        weights = np.array(list(weights.values()))
    
    return SingleModelContext(model, weights, annualize)


def create_active_risk_context(
    portfolio_model: RiskModel,
    benchmark_model: RiskModel,
    portfolio_weights: Union[np.ndarray, Dict[str, float]],
    benchmark_weights: Union[np.ndarray, Dict[str, float]],
    annualize: bool = True
) -> MultiModelContext:
    """
    Helper function to create MultiModelContext (for backwards compatibility).
    
    Note: It's better to use analyze_active_risk() directly.
    """
    warnings.warn(
        "create_active_risk_context is deprecated. Use analyze_active_risk() directly.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if isinstance(portfolio_weights, dict):
        portfolio_weights = np.array(list(portfolio_weights.values()))
    if isinstance(benchmark_weights, dict):
        benchmark_weights = np.array(list(benchmark_weights.values()))
    
    return MultiModelContext(
        portfolio_model, benchmark_model,
        portfolio_weights, benchmark_weights,
        annualize
    )