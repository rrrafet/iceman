"""
Unified Risk Decomposer - A clean, SOLID-compliant risk decomposition system.

This module provides a single, unified interface for all types of risk decomposition,
replacing the previous fragmented architecture with a clean, extensible design.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import RiskResultSchema

from .context import RiskContext
from .strategies import RiskAnalysisStrategy, StrategyFactory
from .base import RiskDecomposerBase


class RiskDecomposer(RiskDecomposerBase):
    """
    Unified risk decomposer supporting all types of risk analysis.
    
    This class replaces the previous RiskDecomposer, ActiveRiskDecomposer, and
    UnifiedRiskDecomposer classes with a single, clean interface that uses
    dependency injection and the strategy pattern for maximum flexibility.
    
    Key Features:
    - Single interface for all risk decomposition types
    - Type-safe contexts prevent configuration errors
    - Extensible strategy system for new decomposition methods  
    - Comprehensive validation and error handling
    - Full backward compatibility with existing code
    
    Examples
    --------
    # Traditional portfolio risk analysis
    >>> from spark.risk import RiskDecomposer, create_single_model_context
    >>> context = create_single_model_context(model, weights)
    >>> decomposer = RiskDecomposer(context)
    >>> print(f"Portfolio risk: {decomposer.portfolio_volatility:.2%}")
    
    # Active risk analysis
    >>> from spark.risk import create_active_risk_context
    >>> context = create_active_risk_context(port_model, bench_model, port_weights, bench_weights)
    >>> decomposer = RiskDecomposer(context)  
    >>> print(f"Active risk: {decomposer.total_active_risk:.2%}")
    """
    
    def __init__(
        self, 
        context: RiskContext,
        strategy: Optional[RiskAnalysisStrategy] = None
    ):
        """
        Initialize unified risk decomposer.
        
        Parameters
        ----------
        context : RiskContext
            Risk analysis context containing models, weights, and configuration
        strategy : RiskAnalysisStrategy, optional
            Analysis strategy. If None, will be auto-selected based on context type.
        """
        self._context = context
        self._strategy = strategy or StrategyFactory.get_strategy_for_context(context)
        
        # Validate strategy compatibility
        supported_types = self._strategy.get_supported_context_types()
        if not isinstance(context, supported_types):
            raise TypeError(
                f"Strategy {type(self._strategy).__name__} does not support context type {type(context).__name__}. "
                f"Supported types: {[t.__name__ for t in supported_types]}"
            )
        
        # Perform analysis (now returns unified schema)
        self._schema = self._strategy.analyze(context)
        
        # Extract data directly from schema using new methods
        # Determine the lens based on context type
        if hasattr(context, 'analysis_type'):
            lens = context.analysis_type.lower() if context.analysis_type else 'portfolio'
        else:
            lens = 'portfolio'
        
        # Cache commonly accessed properties from schema
        core_metrics = self._schema._data.get(lens, {}).get('core_metrics', {})
        self._portfolio_volatility = core_metrics.get('total_risk', 0.0)
        self._factor_risk_contribution = core_metrics.get('factor_risk_contribution', 0.0)
        self._specific_risk_contribution = core_metrics.get('specific_risk_contribution', 0.0)
        
        # Extract contributions
        contributions = self._schema._data.get(lens, {}).get('contributions', {})
        self._factor_contributions = contributions.get('by_factor', {})
        self._asset_total_contributions = contributions.get('by_asset', {})
        
        # Store weights and matrices for property access
        self._weights = self._schema._data.get('weights', {})
        self._matrices = self._schema._data.get(lens, {}).get('matrices', {})
        
        # Create a results dict for backward compatibility with properties
        self._results = {
            'portfolio_volatility': self._portfolio_volatility,
            'factor_risk_contribution': self._factor_risk_contribution,
            'specific_risk_contribution': self._specific_risk_contribution,
            'factor_contributions': self._factor_contributions,
            'asset_total_contributions': self._asset_total_contributions,
            'portfolio_weights': self._weights.get('portfolio_weights', {}),
            'weighted_betas': self._matrices.get('weighted_betas', {}),
            'asset_by_factor_contributions': self._matrices.get('factor_risk_contributions', {}),
        }
    
    # =========================================================================
    # CORE ABSTRACT PROPERTIES (Required by base class)
    # =========================================================================
    
    @property
    def portfolio_volatility(self) -> float:
        """
        Total portfolio volatility (risk).
        
        For single portfolio analysis, this is the absolute portfolio risk.
        For active risk analysis, this represents the total active risk.
        
        Returns
        -------
        float
            Annualized volatility/risk measure
        """
        return self._portfolio_volatility
    
    @property
    def portfolio_weights(self) -> np.ndarray:
        """
        Portfolio weights as a flat array.
        
        Returns
        -------
        np.ndarray
            Array of portfolio weights (N assets)
        """
        return self._results['portfolio_weights']
    
    @property
    def portfolio_factor_exposure(self) -> np.ndarray:
        """
        Portfolio's total factor exposures.
        
        Calculated as: B^T @ w where B is factor loadings matrix and w is weights.
        
        Returns
        -------
        np.ndarray
            Array of factor exposures (K factors)
        """
        return self._results['portfolio_factor_exposure']
    
    @property
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
        return self._factor_risk_contribution
    
    @property
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
        return self._specific_risk_contribution
    
    @property
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
        return self._asset_total_contributions
    
    @property
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
        return self._factor_contributions
    
    @property
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
        return self._results.get('marginal_factor_contributions', np.zeros(len(self.factor_contributions)))
    
    # =========================================================================
    # EXTENDED PROPERTIES (Available when applicable)
    # =========================================================================
    
    @property
    def marginal_asset_contributions(self) -> np.ndarray:
        """
        Marginal contribution to risk from each asset.
        
        Returns
        -------
        np.ndarray
            Array of marginal asset contributions (N assets, volatility units per unit weight)
        """
        return self._results.get('marginal_asset_contributions', np.zeros(len(self.asset_total_contributions)))
    
    @property
    def percent_total_contributions(self) -> np.ndarray:
        """
        Asset contributions as percentages of total portfolio volatility.
        
        Returns
        -------
        np.ndarray
            Array of percentage contributions (N assets, fraction of total risk)
        """
        return self._results.get('percent_total_contributions', self.asset_total_contributions / self.portfolio_volatility)
    
    @property
    def percent_factor_contributions(self) -> np.ndarray:
        """
        Factor contributions as percentages of total portfolio volatility.
        
        Returns
        -------
        np.ndarray
            Array of percentage factor contributions (K factors, fraction of total risk)
        """
        return self._results.get('percent_factor_contributions', self.factor_contributions / self.portfolio_volatility)
    
    # =========================================================================
    # ACTIVE RISK SPECIFIC PROPERTIES (Available for active risk analysis)
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
        return self._results.get('total_active_risk')
    
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
        return self._results.get('benchmark_weights')
    
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
        return self._results.get('active_weights')
    
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
        return self._results.get('benchmark_factor_exposure')
    
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
        return self._results.get('active_factor_exposure')
    
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
        return self._results.get('allocation_factor_risk')
    
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
        return self._results.get('allocation_specific_risk')
    
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
        return self._results.get('selection_factor_risk')
    
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
        return self._results.get('selection_specific_risk')
    
    @property
    def total_allocation_risk(self) -> Optional[float]:
        """
        Total allocation risk (allocation factor + specific, for active risk decomposition only).
        
        Returns
        -------
        Optional[float]
            Total allocation risk, or None if not applicable
        """
        return self._results.get('total_allocation_risk')
    
    @property
    def total_selection_risk(self) -> Optional[float]:
        """
        Total selection risk (selection factor + specific, for active risk decomposition only).
        
        Returns
        -------
        Optional[float]
            Total selection risk, or None if not applicable
        """
        return self._results.get('total_selection_risk')
    
    # =========================================================================
    # ASSET-LEVEL ACTIVE CONTRIBUTIONS (Available for active risk analysis)
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
        return self._results.get('asset_allocation_factor_contributions')
    
    @property
    def asset_allocation_specific_contributions(self) -> Optional[np.ndarray]:
        """
        Asset-level contributions to allocation specific risk (for active risk decomposition only).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of asset allocation specific contributions, or None if not applicable
        """
        return self._results.get('asset_allocation_specific_contributions')
    
    @property
    def asset_selection_factor_contributions(self) -> Optional[np.ndarray]:
        """
        Asset-level contributions to selection factor risk (for active risk decomposition only).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of asset selection factor contributions, or None if not applicable
        """
        return self._results.get('asset_selection_factor_contributions')
    
    @property
    def asset_selection_specific_contributions(self) -> Optional[np.ndarray]:
        """
        Asset-level contributions to selection specific risk (for active risk decomposition only).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of asset selection specific contributions, or None if not applicable
        """
        return self._results.get('asset_selection_specific_contributions')
    
    # =========================================================================
    # FACTOR-LEVEL ACTIVE CONTRIBUTIONS (Available for active risk analysis)
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
        return self._results.get('factor_allocation_contributions')
    
    @property
    def factor_selection_contributions(self) -> Optional[np.ndarray]:
        """
        Factor-level contributions to selection risk (for active risk decomposition only).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of factor selection contributions, or None if not applicable
        """
        return self._results.get('factor_selection_contributions')
    
    # =========================================================================
    # ADDITIONAL PROPERTIES (For enhanced functionality)
    # =========================================================================
    
    @property
    def risk_decomposition(self) -> Optional[Dict[str, float]]:
        """
        Risk decomposition breakdown (for active risk analysis).
        
        Returns
        -------
        Optional[Dict[str, float]]
            Dictionary with risk component percentages, or None if not applicable
        """
        return self._results.get('risk_decomposition')
    
    @property
    def asset_factor_contributions(self) -> Optional[np.ndarray]:
        """
        Asset-level contributions to factor risk (for single portfolio analysis).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of asset factor contributions, or None if not applicable
        """
        return self._results.get('asset_factor_contributions')
    
    @property
    def asset_specific_contributions(self) -> Optional[np.ndarray]:
        """
        Asset-level contributions to specific risk (for single portfolio analysis).
        
        Returns
        -------
        Optional[np.ndarray]
            Array of asset specific contributions, or None if not applicable
        """
        return self._results.get('asset_specific_contributions')
    
    @property
    def asset_by_factor_contributions(self) -> Optional[np.ndarray]:
        """
        Asset x factor matrix of risk contributions.
        
        This is a detailed breakdown showing how each asset contributes
        to risk through each factor. Use with care as this is non-standard.
        Available only for single portfolio analysis.
        
        Returns
        -------
        Optional[np.ndarray]
            Matrix of contributions (N assets x K factors, volatility units), or None if not applicable
        """
        return self._results.get('asset_by_factor_contributions')
    
    # =========================================================================
    # VALIDATION METHODS (Required by base class)
    # =========================================================================
    
    def validate_contributions(self, tolerance: float = 1e-6) -> Dict:
        """
        Validate that risk decomposition results are mathematically consistent.
        
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
        validation = self._results.get('validation', {})
        
        # Add additional validation checks
        results = validation.copy() if validation else {}
        
        # Get names for enhanced reporting
        asset_names = self._results.get('asset_names', [])
        factor_names = self._results.get('factor_names', [])
        
        # Asset contributions sum check (if not already included)
        if 'asset_sum_check' not in results:
            asset_sum = np.sum(self.asset_total_contributions)
            results['asset_sum_check'] = {
                'expected': self.portfolio_volatility,
                'actual': asset_sum,
                'difference': abs(asset_sum - self.portfolio_volatility),
                'passes': abs(asset_sum - self.portfolio_volatility) < tolerance,
                'description': f'Sum of {len(self.asset_total_contributions)} asset contributions equals total portfolio risk'
            }
        
        # Factor + specific sum check (if not already included)
        if 'factor_specific_sum_check' not in results:
            factor_specific_sum = self.factor_risk_contribution + self.specific_risk_contribution
            results['factor_specific_sum_check'] = {
                'expected': self.portfolio_volatility,
                'actual': factor_specific_sum,
                'difference': abs(factor_specific_sum - self.portfolio_volatility),
                'passes': abs(factor_specific_sum - self.portfolio_volatility) < tolerance,
                'description': f'Factor risk + specific risk equals total portfolio risk'
            }
        
        # Enhanced validation: Individual asset contribution reasonableness
        if 'asset_contribution_reasonableness' not in results:
            max_asset_contrib = np.max(np.abs(self.asset_total_contributions))
            max_weight = np.max(self.portfolio_weights)
            
            # Find the asset with maximum contribution
            max_contrib_idx = np.argmax(np.abs(self.asset_total_contributions))
            max_contrib_name = asset_names[max_contrib_idx] if len(asset_names) > max_contrib_idx else f"asset_{max_contrib_idx}"
            
            results['asset_contribution_reasonableness'] = {
                'max_contribution': max_asset_contrib,
                'max_contribution_asset': max_contrib_name,
                'max_weight': max_weight,
                'max_contrib_percentage': 100.0 * max_asset_contrib / self.portfolio_volatility if self.portfolio_volatility > 0 else 0.0,
                'passes': max_asset_contrib <= self.portfolio_volatility,  # No single asset should contribute more than total risk
                'description': f'Maximum individual asset contribution is reasonable (≤ total risk)'
            }
        
        # Enhanced validation: Factor contribution reasonableness
        if 'factor_contribution_reasonableness' not in results and len(self.factor_contributions) > 0:
            factor_sum = np.sum(self.factor_contributions)
            max_factor_contrib = np.max(np.abs(self.factor_contributions))
            
            # Find the factor with maximum contribution
            max_factor_idx = np.argmax(np.abs(self.factor_contributions))
            max_factor_name = factor_names[max_factor_idx] if len(factor_names) > max_factor_idx else f"factor_{max_factor_idx}"
            
            results['factor_contribution_reasonableness'] = {
                'factor_sum': factor_sum,
                'max_factor_contribution': max_factor_contrib,
                'max_factor_name': max_factor_name,
                'max_factor_percentage': 100.0 * max_factor_contrib / self.portfolio_volatility if self.portfolio_volatility > 0 else 0.0,
                'passes': abs(factor_sum - self.factor_risk_contribution) < tolerance,
                'description': f'Sum of individual factor contributions equals total factor risk'
            }
        
        # Overall validation
        all_checks_pass = all(result.get('passes', False) for key, result in results.items() 
                             if key != 'overall_validation' and isinstance(result, dict))
        results['overall_validation'] = {
            'passes': all_checks_pass,
            'message': 'All validations passed' if all_checks_pass else 'Some validations failed',
            'total_checks': len([k for k in results.keys() if k != 'overall_validation']),
            'passed_checks': len([k for k, v in results.items() if k != 'overall_validation' and isinstance(v, dict) and v.get('passes', False)])
        }
        
        return results
    
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
        results = self.validate_contributions(tolerance)
        
        lines = []
        analysis_type = self._results.get('analysis_type', 'unknown')
        lines.append(f"=== {analysis_type.title()} Risk Decomposition Validation ===")
        lines.append(f"Total Risk: {self.portfolio_volatility:.6f}")
        lines.append("")
        
        for check_name, check_result in results.items():
            if check_name == 'overall_validation' or not isinstance(check_result, dict):
                continue
                
            status = "✓ PASS" if check_result.get('passes', False) else "✗ FAIL"
            lines.append(f"{check_name}: {status}")
            if 'expected' in check_result:
                lines.append(f"  Expected: {check_result['expected']:.6f}")
                lines.append(f"  Actual:   {check_result['actual']:.6f}")
                lines.append(f"  Diff:     {check_result['difference']:.6f}")
            lines.append("")
        
        lines.append(f"Overall: {results['overall_validation']['message']}")
        return "\\n".join(lines)
    
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
        summary = {
            'analysis_type': self._results.get('analysis_type', 'unknown'),
            'portfolio_volatility': self.portfolio_volatility,
            'factor_risk_contribution': self.factor_risk_contribution,
            'specific_risk_contribution': self.specific_risk_contribution,
            'factor_risk_percentage': 100.0 * self.factor_risk_contribution / self.portfolio_volatility if self.portfolio_volatility > 0 else 0.0,
            'specific_risk_percentage': 100.0 * self.specific_risk_contribution / self.portfolio_volatility if self.portfolio_volatility > 0 else 0.0,
            'number_of_assets': len(self.portfolio_weights),
            'number_of_factors': len(self.factor_contributions),
            'annualized': self._results.get('annualized', True),
            'frequency': self._results.get('frequency', 'unknown'),
            'asset_names': self._results.get('asset_names', []),
            'factor_names': self._results.get('factor_names', []),
            'validation_results': self.validate_contributions()
        }
        
        # Add active risk specific components if available
        if self.total_active_risk is not None:
            summary.update({
                'total_active_risk': self.total_active_risk,
                'allocation_factor_risk': self.allocation_factor_risk,
                'allocation_specific_risk': self.allocation_specific_risk,
                'selection_factor_risk': self.selection_factor_risk,
                'selection_specific_risk': self.selection_specific_risk,
                'total_allocation_risk': self.total_allocation_risk,
                'total_selection_risk': self.total_selection_risk,
            })
            
            # Calculate percentages
            if self.total_active_risk > 0:
                summary.update({
                    'allocation_factor_percentage': 100.0 * (self.allocation_factor_risk or 0) / self.total_active_risk,
                    'allocation_specific_percentage': 100.0 * (self.allocation_specific_risk or 0) / self.total_active_risk,
                    'selection_factor_percentage': 100.0 * (self.selection_factor_risk or 0) / self.total_active_risk,
                    'selection_specific_percentage': 100.0 * (self.selection_specific_risk or 0) / self.total_active_risk,
                })
        
        # Add top contributors with names
        asset_names = self._results.get('asset_names', [])
        factor_names = self._results.get('factor_names', [])
        
        summary.update({
            'top_asset_contributors': self._get_top_contributors_with_names(
                self.asset_total_contributions, asset_names, 5, "asset"
            ),
            'top_factor_contributors': self._get_top_contributors_with_names(
                self.factor_contributions, factor_names, 5, "factor"
            )
        })
        
        return summary
    
    def _get_top_contributors(self, contributions: np.ndarray, n: int = 5) -> Dict:
        """
        Get top N contributors from a contributions array.
        
        Parameters
        ----------
        contributions : np.ndarray
            Array of contributions
        n : int, default 5
            Number of top contributors to return
            
        Returns
        -------
        Dict
            Dictionary with indices, values, and percentages of top contributors
        """
        if len(contributions) == 0:
            return {'indices': [], 'values': [], 'percentages': []}
        
        abs_contributions = np.abs(contributions)
        top_indices = np.argsort(abs_contributions)[-n:][::-1]
        
        return {
            'indices': top_indices.tolist(),
            'values': contributions[top_indices].tolist(),
            'percentages': (100.0 * contributions[top_indices] / self.portfolio_volatility).tolist() if self.portfolio_volatility > 0 else [0.0] * len(top_indices)
        }
    
    def _get_beta_matrix_as_list(self):
        """
        Get the beta/factor loadings matrix as a list for JSON serialization.
        
        Returns
        -------
        list or None
            Beta matrix as nested list (N x K) or None if not available
        """
        try:
            if hasattr(self._context, 'model') and hasattr(self._context.model, 'beta'):
                return self._context.model.beta.tolist()
        except (AttributeError, IndexError):
            pass
        return None
    
    def _get_top_contributors_with_names(
        self, 
        contributions: np.ndarray, 
        names: list, 
        n: int = 5, 
        fallback_prefix: str = "item"
    ) -> Dict:
        """
        Get top N contributors from a contributions array with proper names.
        
        Parameters
        ----------
        contributions : np.ndarray
            Array of contributions
        names : list
            List of names corresponding to contributions
        n : int, default 5
            Number of top contributors to return
        fallback_prefix : str, default "item"
            Prefix for fallback names when names list is insufficient
            
        Returns
        -------
        Dict
            Dictionary with indices, names, values, and percentages of top contributors
        """
        if len(contributions) == 0:
            return {
                'indices': [], 
                'names': [], 
                'values': [], 
                'percentages': [],
                'named_contributions': {}
            }
        
        abs_contributions = np.abs(contributions)
        top_indices = np.argsort(abs_contributions)[-n:][::-1]
        
        # Get names for top contributors
        if len(names) == len(contributions):
            top_names = [names[i] for i in top_indices]
        else:
            top_names = [f"{fallback_prefix}_{i}" for i in top_indices]
        
        top_values = contributions[top_indices].tolist()
        top_percentages = (100.0 * contributions[top_indices] / self.portfolio_volatility).tolist() if self.portfolio_volatility > 0 else [0.0] * len(top_indices)
        
        # Create named dictionary for easy lookup
        named_contributions = {name: value for name, value in zip(top_names, top_values)}
        
        return {
            'indices': top_indices.tolist(),
            'names': top_names,
            'values': top_values,
            'percentages': top_percentages,
            'named_contributions': named_contributions
        }
    
    # =========================================================================
    # CONTEXT AND STRATEGY ACCESS
    # =========================================================================
    
    @property
    def context(self) -> RiskContext:
        """Get the risk analysis context"""
        return self._context
    
    @property
    def strategy(self) -> RiskAnalysisStrategy:
        """Get the risk analysis strategy"""
        return self._strategy
    
    @property
    def results(self) -> Dict[str, Any]:
        """Get the full results dictionary"""
        return self._results.copy()
    
    def is_active_decomposer(self) -> bool:
        """
        Check if this is an active risk decomposer.
        
        Returns
        -------
        bool
            True if this decomposer supports active risk analysis
        """
        return self.total_active_risk is not None
    
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
    
    def get_risk_summary_table(self) -> pd.DataFrame:
        """
        Generate a formatted DataFrame summarizing key risk metrics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with risk component names and values
        """
        data = {
            'Risk Component': ['Total Risk', 'Factor Risk', 'Specific Risk'],
            'Value': [
                self.portfolio_volatility,
                self.factor_risk_contribution,
                self.specific_risk_contribution
            ],
            'Percentage': [
                100.0,
                100.0 * self.factor_risk_contribution / self.portfolio_volatility if self.portfolio_volatility > 0 else 0.0,
                100.0 * self.specific_risk_contribution / self.portfolio_volatility if self.portfolio_volatility > 0 else 0.0
            ]
        }
        
        # Add active risk components if available
        if self.is_active_decomposer():
            active_data = {
                'Risk Component': [
                    'Allocation Factor Risk',
                    'Allocation Specific Risk', 
                    'Selection Factor Risk',
                    'Selection Specific Risk'
                ],
                'Value': [
                    self.allocation_factor_risk or 0.0,
                    self.allocation_specific_risk or 0.0,
                    self.selection_factor_risk or 0.0,
                    self.selection_specific_risk or 0.0
                ],
                'Percentage': [
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
    
    def to_dict(self, include_arrays: bool = True, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export RiskDecomposer results to unified schema format and return as dictionary.
        
        This is the primary export method that returns results in the standardized
        unified schema format as a dictionary.
        
        Parameters
        ----------
        include_arrays : bool, default True
            Whether to include array properties (contributions, exposures, weights).
        include_metadata : bool, default True
            Whether to include metadata about context, strategy, and validation.
            
        Returns
        -------
        Dict[str, Any]
            Complete risk analysis results in unified schema format.
        """
        # Create unified schema and return as dictionary
        schema = self._create_unified_schema()
        return schema.to_dict()
    
    def to_json(self, include_arrays: bool = True, include_metadata: bool = True, **json_kwargs) -> str:
        """
        Export RiskDecomposer results to JSON string.
        
        Uses the unified schema internally for consistent JSON serialization.
        
        Parameters
        ----------
        include_arrays : bool, default True
            Whether to include array properties (contributions, exposures, weights).
        include_metadata : bool, default True
            Whether to include metadata about context, strategy, and validation.
        **json_kwargs
            Additional keyword arguments passed to json.dumps()
            
        Returns
        -------
        str
            JSON string of complete risk analysis results.
        """
        # Create unified schema and use its JSON export
        schema = self._create_unified_schema()
        return schema.to_json()

    def to_unified_schema(self) -> 'RiskResultSchema':
        """
        Get the unified schema results from the decomposer.
        
        Returns the schema that was created during analysis, providing
        direct access to the unified format results.
        
        Returns
        -------
        RiskResultSchema
            Results in unified schema format
        """
        # Return the schema that was created during analysis
        return self._schema
    
    def _create_unified_schema(self) -> 'RiskResultSchema':
        """
        Internal method to create unified schema - just returns the stored schema.
        
        Returns
        -------
        RiskResultSchema
            Results in unified schema format
        """
        # Return the schema that was created during analysis
        return self._schema
    
    # =========================================================================
    # FACTORY METHODS FOR BACKWARD COMPATIBILITY  
    # =========================================================================
