"""
Risk calculation utilities and core mathematical operations.

This module provides the fundamental risk calculation logic that was previously
duplicated across RiskDecomposer and ActiveRiskDecomposer classes.
"""

import numpy as np
import warnings
from typing import Optional, Dict, Any


class RiskCalculator:
    """
    Core risk calculation engine providing mathematical operations for portfolio risk analysis.
    
    This class encapsulates all the mathematical operations needed for risk decomposition,
    eliminating code duplication and providing a single source of truth for calculations.
    
    Key principles:
    - Pure mathematical functions with no side effects
    - Consistent input/output interfaces
    - Proper error handling and validation
    - Support for both single and multi-model scenarios
    """
    
    @staticmethod
    def calculate_portfolio_variance(
        covar: np.ndarray, 
        weights: np.ndarray
    ) -> float:
        """
        Calculate portfolio variance: w^T * Σ * w
        
        Parameters
        ----------
        covar : np.ndarray
            Covariance matrix (N x N)
        weights : np.ndarray
            Portfolio weights (N,)
            
        Returns
        -------
        float
            Portfolio variance (non-annualized)
        """
        weights = np.asarray(weights).flatten()
        if weights.shape[0] != covar.shape[0]:
            raise ValueError(f"Weight dimension {weights.shape[0]} doesn't match covariance dimension {covar.shape[0]}")
        
        return float(weights.T @ covar @ weights)
    
    @staticmethod
    def calculate_portfolio_volatility(
        covar: np.ndarray, 
        weights: np.ndarray
    ) -> float:
        """
        Calculate portfolio volatility: sqrt(w^T * Σ * w)
        
        Parameters
        ----------
        covar : np.ndarray
            Covariance matrix (N x N)
        weights : np.ndarray
            Portfolio weights (N,)
            
        Returns
        -------
        float
            Portfolio volatility (raw, non-annualized)
        """
        variance = RiskCalculator.calculate_portfolio_variance(covar, weights)
        return np.sqrt(variance)
    
    @staticmethod
    def calculate_factor_exposures(
        beta: np.ndarray, 
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Calculate portfolio factor exposures: B^T * w
        
        Parameters
        ----------
        beta : np.ndarray
            Factor loadings matrix (N x K)
        weights : np.ndarray
            Portfolio weights (N,)
            
        Returns
        -------
        np.ndarray
            Factor exposures (K,)
        """
        weights = np.asarray(weights).reshape(-1, 1)
        if weights.shape[0] != beta.shape[0]:
            raise ValueError(f"Weight dimension {weights.shape[0]} doesn't match beta dimension {beta.shape[0]}")
        
        return (beta.T @ weights).flatten()
    
    @staticmethod
    def calculate_factor_risk(
        beta: np.ndarray,
        factor_covar: np.ndarray, 
        weights: np.ndarray
    ) -> float:
        """
        Calculate factor risk contribution: sqrt(f^T * Ω * f) where f = B^T * w
        
        Parameters
        ----------
        beta : np.ndarray
            Factor loadings matrix (N x K)
        factor_covar : np.ndarray
            Factor covariance matrix (K x K)
        weights : np.ndarray
            Portfolio weights (N,)
            
        Returns
        -------
        float
            Factor risk contribution (raw, non-annualized)
        """
        factor_exposures = RiskCalculator.calculate_factor_exposures(beta, weights)
        factor_variance = factor_exposures.T @ factor_covar @ factor_exposures
        return np.sqrt(factor_variance)
    
    @staticmethod
    def calculate_specific_risk(
        resvar: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        Calculate specific risk contribution: sqrt(w^T * D * w)
        
        Parameters
        ----------
        resvar : np.ndarray
            Residual variance matrix (N x N) or vector (N,)
        weights : np.ndarray
            Portfolio weights (N,)
            
        Returns
        -------
        float
            Specific risk contribution (raw, non-annualized)
        """
        weights = np.asarray(weights).flatten()
        
        # Handle both matrix and vector formats
        if resvar.ndim == 1:
            specific_variance = weights.T @ (resvar * weights)
        else:
            specific_variance = weights.T @ resvar @ weights
            
        return np.sqrt(specific_variance)
    
    @staticmethod
    def calculate_marginal_contributions(
        covar: np.ndarray,
        weights: np.ndarray,
        portfolio_volatility: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate marginal risk contributions: (Σ * w) / σ_p
        
        Parameters
        ----------
        covar : np.ndarray
            Covariance matrix (N x N)
        weights : np.ndarray
            Portfolio weights (N,)
        portfolio_volatility : float, optional
            Pre-calculated portfolio volatility. If None, will be calculated.
            
        Returns
        -------
        np.ndarray
            Marginal contributions (raw, non-annualized)
        """
        weights = np.asarray(weights).reshape(-1, 1)
        
        if portfolio_volatility is None:
            portfolio_volatility = RiskCalculator.calculate_portfolio_volatility(covar, weights.flatten())
        
        if portfolio_volatility == 0:
            return np.zeros(weights.shape[0])
        
        return (covar @ weights).flatten() / portfolio_volatility
    
    @staticmethod
    def calculate_total_contributions(
        marginal_contributions: np.ndarray,
        weights: np.ndarray,
        expected_total: Optional[float] = None,
        tolerance: float = 1e-4
    ) -> np.ndarray:
        """
        Calculate total risk contributions: w * MCTR
        
        Parameters
        ----------
        marginal_contributions : np.ndarray
            Marginal contributions (N,)
        weights : np.ndarray
            Portfolio weights (N,)
        expected_total : float, optional
            Expected sum for validation
        tolerance : float, default 1e-4
            Tolerance for validation warning
            
        Returns
        -------
        np.ndarray
            Total contributions (N,)
        """
        weights = np.asarray(weights).flatten()
        contributions = weights * marginal_contributions
        
        if expected_total is not None:
            actual_total = np.sum(contributions)
            if not np.isclose(actual_total, expected_total, rtol=tolerance):
                warnings.warn(
                    f"Sum of contributions ({actual_total:.6f}) does not match "
                    f"expected total ({expected_total:.6f}). Difference: {abs(actual_total - expected_total):.6f}"
                )
        
        return contributions
    
    @staticmethod
    def calculate_marginal_factor_contributions(
        beta: np.ndarray,
        factor_covar: np.ndarray,
        weights: np.ndarray,
        portfolio_volatility: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate marginal factor contributions: f * MFCTR
        
        Parameters
        ----------
        beta : np.ndarray
            Factor loadings matrix (N x K)
        factor_covar : np.ndarray
            Factor covariance matrix (K x K)
        weights : np.ndarray
            Portfolio weights (N,)
        portfolio_volatility : float, optional
            Pre-calculated portfolio volatility. If None, will be calculated.
            
        Returns
        -------
        np.ndarray
            Marginal factor contributions (raw, non-annualized)
        """
        factor_exposures = RiskCalculator.calculate_factor_exposures(beta, weights)
        
        if portfolio_volatility is None:
            # Calculate portfolio volatility from full covariance matrix
            full_covar = beta @ factor_covar @ beta.T
            portfolio_volatility = RiskCalculator.calculate_portfolio_volatility(full_covar, weights)
        
        if portfolio_volatility == 0:
            return np.zeros(factor_exposures.shape[0])
        
        return (factor_covar @ factor_exposures) / portfolio_volatility
    
    @staticmethod
    def calculate_factor_contributions(
        beta: np.ndarray,
        factor_covar: np.ndarray,
        weights: np.ndarray,
        portfolio_volatility: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate individual factor contributions: f * MFCTR
        
        Parameters
        ----------
        beta : np.ndarray
            Factor loadings matrix (N x K)
        factor_covar : np.ndarray
            Factor covariance matrix (K x K)
        weights : np.ndarray
            Portfolio weights (N,)
        portfolio_volatility : float, optional
            Pre-calculated portfolio volatility. If None, will be calculated.
            
        Returns
        -------
        np.ndarray
            Factor contributions (raw, non-annualized)
        """
        factor_exposures = RiskCalculator.calculate_factor_exposures(beta, weights)
        
        if portfolio_volatility is None:
            # Calculate portfolio volatility from full covariance matrix
            full_covar = beta @ factor_covar @ beta.T
            portfolio_volatility = RiskCalculator.calculate_portfolio_volatility(full_covar, weights)
        
        if portfolio_volatility == 0:
            return np.zeros(factor_exposures.shape[0])
        
        marginal_factor = (factor_covar @ factor_exposures) / portfolio_volatility
        return factor_exposures * marginal_factor
    
    @staticmethod
    def calculate_specific_contributions(
        resvar: np.ndarray,
        weights: np.ndarray,
        portfolio_volatility: float
    ) -> np.ndarray:
        """
        Calculate specific risk contributions: w * SCR
        
        Parameters
        ----------
        resvar : np.ndarray
            Residual variance matrix (N x N) or vector (N,)
        weights : np.ndarray
            Portfolio weights (N,)
        portfolio_volatility : float
            Pre-calculated portfolio volatility
            
        Returns
        -------
        np.ndarray
            Specific risk contributions (raw, non-annualized)
        """
        weights = np.asarray(weights).flatten()
        
        if portfolio_volatility == 0:
            return np.zeros(weights.shape[0])
        
        if resvar.ndim == 1:
            contributions = weights * (resvar * weights) / portfolio_volatility
        else:
            contributions = weights * (resvar @ weights) / portfolio_volatility
        
        return contributions

    @staticmethod
    def calculate_active_risk(
        benchmark_covar: np.ndarray,
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray,
        active_covar: Optional[np.ndarray] = None,
        cross_covar: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate active risk using appropriate covariance matrices including cross-correlation.
        
        The complete active risk formula is:
        σ²_TE = d^T Σ_b d + w^T Ω w + 2 d^T C w
        
        where:
        - d = active weights (portfolio - benchmark)
        - w = portfolio weights  
        - Σ_b = benchmark covariance matrix
        - Ω = active covariance matrix
        - C = cross-covariance matrix between benchmark and active returns
        
        Parameters
        ----------
        benchmark_covar : np.ndarray
            Benchmark covariance matrix (Σ_b)
        portfolio_weights : np.ndarray
            Portfolio weights (w)
        benchmark_weights : np.ndarray
            Benchmark weights (b)
        active_covar : np.ndarray, optional
            Active covariance matrix (Ω). If None, approximated from portfolio and benchmark.
        cross_covar : np.ndarray, optional
            Cross-covariance matrix between benchmark and active returns (C). If None, 
            cross-correlation term is assumed to be zero.
            
        Returns
        -------
        float
            Active risk (raw, non-annualized tracking error volatility)
        """
        portfolio_weights = np.asarray(portfolio_weights).flatten()
        benchmark_weights = np.asarray(benchmark_weights).flatten()
        active_weights = portfolio_weights - benchmark_weights
        
        if active_covar is not None:
            # Benchmark tilt component: d^T Σ_b d
            benchmark_component = RiskCalculator.calculate_portfolio_variance(benchmark_covar, active_weights)
            
            # Active selection component: w^T Ω w  
            active_component = RiskCalculator.calculate_portfolio_variance(active_covar, portfolio_weights)
            
            # Cross-correlation component: 2 d^T C w
            cross_component = 0.0
            if cross_covar is not None:
                cross_component = 2.0 * active_weights.T @ cross_covar @ portfolio_weights
            
            active_variance = benchmark_component + active_component + cross_component
        else:
            raise ValueError("Active covariance matrix must be provided for active risk calculation.")
        
        return np.sqrt(max(0.0, active_variance))  # Ensure non-negative variance
    
    @staticmethod
    def calculate_cross_covariance(
        benchmark_returns: np.ndarray,
        active_returns: np.ndarray,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate cross-covariance matrix C = Cov(R_b, α) between benchmark and active returns.
        
        Parameters
        ----------
        benchmark_returns : np.ndarray
            Benchmark returns matrix (T x N) where T is time periods, N is assets
        active_returns : np.ndarray
            Active returns matrix (T x N) representing alpha/residual returns
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Cross-covariance matrix C (N x N)
        """
        if validate:
            benchmark_returns = np.asarray(benchmark_returns)
            active_returns = np.asarray(active_returns)
            
            if benchmark_returns.shape != active_returns.shape:
                raise ValueError(f"Benchmark returns shape {benchmark_returns.shape} must match active returns shape {active_returns.shape}")
            
            if not np.all(np.isfinite(benchmark_returns)):
                raise ValueError("All benchmark returns must be finite")
            if not np.all(np.isfinite(active_returns)):
                raise ValueError("All active returns must be finite")
        
        # Calculate cross-covariance: Cov(R_b, α) = E[(R_b - μ_b)(α - μ_α)^T]
        benchmark_centered = benchmark_returns - np.mean(benchmark_returns, axis=0, keepdims=True)
        active_centered = active_returns - np.mean(active_returns, axis=0, keepdims=True)
        
        T = benchmark_returns.shape[0]
        cross_covar = (benchmark_centered.T @ active_centered) / (T - 1)
        
        return cross_covar
    
    @staticmethod
    def calculate_cross_variance_contribution(
        cross_covar: np.ndarray,
        active_weights: np.ndarray,
        portfolio_weights: np.ndarray,
        validate: bool = True
    ) -> float:
        """
        Calculate the cross-correlation variance contribution: 2 d^T C w
        
        Parameters
        ----------
        cross_covar : np.ndarray
            Cross-covariance matrix C (N x N)
        active_weights : np.ndarray
            Active weights d = portfolio - benchmark (N,)
        portfolio_weights : np.ndarray
            Portfolio weights w (N,)
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        float
            Cross-correlation variance contribution (raw, non-annualized)
        """
        if validate:
            active_weights = np.asarray(active_weights).flatten()
            portfolio_weights = np.asarray(portfolio_weights).flatten()
            
            if active_weights.shape[0] != cross_covar.shape[0]:
                raise ValueError(f"Active weights length {active_weights.shape[0]} must match cross-covariance dimension {cross_covar.shape[0]}")
            if portfolio_weights.shape[0] != cross_covar.shape[1]:
                raise ValueError(f"Portfolio weights length {portfolio_weights.shape[0]} must match cross-covariance dimension {cross_covar.shape[1]}")
            
            if not np.all(np.isfinite(cross_covar)):
                raise ValueError("All cross-covariance values must be finite")
            if not np.all(np.isfinite(active_weights)):
                raise ValueError("All active weights must be finite")
            if not np.all(np.isfinite(portfolio_weights)):
                raise ValueError("All portfolio weights must be finite")
        
        return 2.0 * active_weights.T @ cross_covar @ portfolio_weights
    
    @staticmethod
    def calculate_marginal_cross_contributions(
        cross_covar: np.ndarray,
        active_weights: np.ndarray,
        portfolio_weights: np.ndarray,
        tracking_error: float,
        validate: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate marginal contributions to risk from cross-correlation term.
        
        Based on gradients:
        - ∇_d σ = (Σ_b d + C w) / σ  (gradient w.r.t. active weights)
        - ∇_w σ = (Ω w + C^T d) / σ  (gradient w.r.t. portfolio weights)
        
        Parameters
        ----------
        cross_covar : np.ndarray
            Cross-covariance matrix C (N x N)
        active_weights : np.ndarray
            Active weights d (N,)
        portfolio_weights : np.ndarray
            Portfolio weights w (N,)
        tracking_error : float
            Current tracking error volatility σ
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (marginal_active, marginal_portfolio) - marginal contributions from cross-term only
        """
        if validate:
            active_weights = np.asarray(active_weights).flatten()
            portfolio_weights = np.asarray(portfolio_weights).flatten()
            
            if active_weights.shape[0] != cross_covar.shape[0]:
                raise ValueError(f"Active weights length must match cross-covariance rows")
            if portfolio_weights.shape[0] != cross_covar.shape[1]:
                raise ValueError(f"Portfolio weights length must match cross-covariance columns")
            
            if not np.isfinite(tracking_error) or tracking_error < 0:
                raise ValueError(f"Tracking error must be finite and non-negative, got {tracking_error}")
            
            if not np.all(np.isfinite(cross_covar)):
                raise ValueError("All cross-covariance values must be finite")
            if not np.all(np.isfinite(active_weights)):
                raise ValueError("All active weights must be finite")
            if not np.all(np.isfinite(portfolio_weights)):
                raise ValueError("All portfolio weights must be finite")
        
        if tracking_error == 0:
            return np.zeros_like(active_weights), np.zeros_like(portfolio_weights)
        
        # Cross-term component of gradient w.r.t. active weights: C w / σ
        marginal_active = (cross_covar @ portfolio_weights) / tracking_error
        
        # Cross-term component of gradient w.r.t. portfolio weights: C^T d / σ
        marginal_portfolio = (cross_covar.T @ active_weights) / tracking_error
        
        return marginal_active, marginal_portfolio
    
    @staticmethod
    def calculate_euler_cross_contributions(
        cross_covar: np.ndarray,
        active_weights: np.ndarray,
        portfolio_weights: np.ndarray,
        tracking_error: float,
        validate: bool = True
    ) -> float:
        """
        Calculate Euler cross-correlation risk contribution: RC_cross = 2 d^T C w / σ
        
        Parameters
        ----------
        cross_covar : np.ndarray
            Cross-covariance matrix C (N x N)
        active_weights : np.ndarray
            Active weights d (N,)
        portfolio_weights : np.ndarray
            Portfolio weights w (N,)
        tracking_error : float
            Current tracking error volatility σ
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        float
            Cross-correlation Euler risk contribution (raw, non-annualized)
        """
        if validate:
            active_weights = np.asarray(active_weights).flatten()
            portfolio_weights = np.asarray(portfolio_weights).flatten()
            
            if not np.isfinite(tracking_error) or tracking_error < 0:
                raise ValueError(f"Tracking error must be finite and non-negative, got {tracking_error}")
        
        if tracking_error == 0:
            return 0.0
        
        cross_variance = RiskCalculator.calculate_cross_variance_contribution(
            cross_covar, active_weights, portfolio_weights, validate
        )
        
        return cross_variance / tracking_error
    
    @staticmethod
    def calculate_asset_level_cross_contributions(
        cross_covar: np.ndarray,
        active_weights: np.ndarray,
        portfolio_weights: np.ndarray,
        tracking_error: float,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate asset-level cross-correlation risk contributions.
        
        Following the cross-term decomposition formula:
        RC_cross,i = (1/σ) * [d_i * (C w)_i + w_i * (C^T d)_i]
        
        This provides a symmetric split of the cross-term contribution across 
        active weights and portfolio weights for each asset.
        
        Parameters
        ----------
        cross_covar : np.ndarray
            Cross-covariance matrix C (N x N)
        active_weights : np.ndarray
            Active weights d (N,)
        portfolio_weights : np.ndarray
            Portfolio weights w (N,)
        tracking_error : float
            Current tracking error volatility σ
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Asset-level cross-correlation contributions (N,)
        """
        if validate:
            active_weights = np.asarray(active_weights).flatten()
            portfolio_weights = np.asarray(portfolio_weights).flatten()
            
            if not np.isfinite(tracking_error) or tracking_error < 0:
                raise ValueError(f"Tracking error must be finite and non-negative, got {tracking_error}")
        
        if tracking_error == 0:
            return np.zeros_like(active_weights)
        
        # Asset-level cross contributions: d_i * (C w)_i + w_i * (C^T d)_i
        cross_active_term = active_weights * (cross_covar @ portfolio_weights)
        cross_portfolio_term = portfolio_weights * (cross_covar.T @ active_weights)
        
        return (cross_active_term + cross_portfolio_term) / tracking_error
    
    @staticmethod
    def validate_risk_decomposition(
        total_risk: float,
        component_risks: Dict[str, float],
        tolerance: float = 1e-6,
        include_cross_correlation: bool = True
    ) -> Dict[str, Any]:
        """
        Validate that risk components sum to total risk including cross-correlation terms.
        
        Enhanced to properly validate Euler identity when cross-correlation terms are present.
        The validation ensures that all components (allocation, selection, cross-correlation)
        sum to the total active risk.
        
        Parameters
        ----------
        total_risk : float
            Total portfolio risk
        component_risks : Dict[str, float]
            Dictionary of risk component names and values
        tolerance : float, default 1e-6
            Numerical tolerance for validation
        include_cross_correlation : bool, default True
            Whether to expect cross-correlation terms in the validation
            
        Returns
        -------
        Dict[str, Any]
            Validation results with pass/fail status and diagnostics
        """
        component_sum = sum(component_risks.values())
        difference = abs(component_sum - total_risk)
        passes = difference < tolerance
        
        # Enhanced diagnostics for cross-correlation decomposition
        diagnostics = {
            'passes': passes,
            'total_risk': total_risk,
            'component_sum': component_sum,
            'difference': difference,
            'tolerance': tolerance,
            'components': component_risks.copy(),
            'message': 'Validation passed' if passes else f'Validation failed: difference = {difference:.6f}'
        }
        
        # Additional validation for cross-correlation scenarios
        if include_cross_correlation:
            # Check for expected cross-correlation components
            cross_components = {k: v for k, v in component_risks.items() if 'cross' in k.lower()}
            traditional_components = {k: v for k, v in component_risks.items() if 'cross' not in k.lower()}
            
            diagnostics.update({
                'has_cross_correlation': len(cross_components) > 0,
                'cross_correlation_components': cross_components,
                'traditional_components': traditional_components,
                'cross_correlation_sum': sum(cross_components.values()) if cross_components else 0.0,
                'traditional_components_sum': sum(traditional_components.values())
            })
            
            # Validate that cross-correlation terms are properly included
            if len(cross_components) == 0 and not passes:
                diagnostics['potential_issue'] = 'Missing cross-correlation terms may be causing Euler identity failure'
            elif len(cross_components) > 0:
                diagnostics['cross_correlation_percentage'] = (sum(cross_components.values()) / total_risk * 100) if total_risk != 0 else 0.0
        
        return diagnostics
    
    @staticmethod
    def validate_active_risk_euler_identity(
        total_active_risk: float,
        allocation_factor_risk: float,
        allocation_specific_risk: float,
        selection_factor_risk: float,
        selection_specific_risk: float,
        cross_correlation_contribution: float = 0.0,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Validate Euler identity for active risk decomposition including cross-correlation.
        
        This specialized validation ensures that the complete active risk formula holds:
        σ_TE = sqrt(σ²_allocation_factor + σ²_allocation_specific + σ²_selection_factor + σ²_selection_specific + σ²_cross)
        
        And that Euler contributions sum correctly:
        RC_total = RC_allocation_factor + RC_allocation_specific + RC_selection_factor + RC_selection_specific + RC_cross
        
        Parameters
        ----------
        total_active_risk : float
            Total active risk (volatility)
        allocation_factor_risk : float
            Allocation factor risk component
        allocation_specific_risk : float
            Allocation specific risk component
        selection_factor_risk : float
            Selection factor risk component
        selection_specific_risk : float
            Selection specific risk component
        cross_correlation_contribution : float, default 0.0
            Cross-correlation Euler contribution
        tolerance : float, default 1e-6
            Numerical tolerance for validation
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive validation results with detailed diagnostics
        """
        # Traditional component risks (without cross-correlation)
        traditional_risk_squared = (allocation_factor_risk ** 2 + allocation_specific_risk ** 2 + 
                                  selection_factor_risk ** 2 + selection_specific_risk ** 2)
        traditional_risk = np.sqrt(traditional_risk_squared)
        
        # Euler contributions validation
        euler_sum = (allocation_factor_risk + allocation_specific_risk + 
                    selection_factor_risk + selection_specific_risk + cross_correlation_contribution)
        
        # Differences
        risk_difference = abs(total_active_risk - traditional_risk)
        euler_difference = abs(euler_sum - total_active_risk)
        
        # Check if cross-correlation is needed
        cross_needed = risk_difference > tolerance
        euler_passes = euler_difference < tolerance
        
        return {
            'euler_identity_passes': euler_passes,
            'total_active_risk': total_active_risk,
            'euler_contributions_sum': euler_sum,
            'euler_difference': euler_difference,
            'traditional_risk_without_cross': traditional_risk,
            'risk_difference_without_cross': risk_difference,
            'cross_correlation_needed': cross_needed,
            'cross_correlation_contribution': cross_correlation_contribution,
            'components': {
                'allocation_factor': allocation_factor_risk,
                'allocation_specific': allocation_specific_risk,
                'selection_factor': selection_factor_risk,
                'selection_specific': selection_specific_risk,
                'cross_correlation': cross_correlation_contribution
            },
            'tolerance': tolerance,
            'message': 'Euler identity validation passed' if euler_passes else f'Euler identity validation failed: difference = {euler_difference:.6f}',
            'cross_correlation_percentage': (cross_correlation_contribution / total_active_risk * 100) if total_active_risk != 0 else 0.0
        }
    
    @staticmethod
    def calculate_percent_contributions(
        contributions: np.ndarray,
        total_risk: float,
        validate: bool = True,
        tolerance: float = 1e-12
    ) -> np.ndarray:
        """
        Calculate percentage contributions normalized by total risk.
        
        Parameters
        ----------
        contributions : np.ndarray
            Raw contributions (N,)
        total_risk : float
            Total risk for normalization
        validate : bool, default True
            Whether to validate inputs
        tolerance : float, default 1e-12
            Minimum total_risk to avoid division by zero
            
        Returns
        -------
        np.ndarray
            Percentage contributions (N,)
        """
        contributions = np.asarray(contributions)
        
        if validate:
            if not np.isfinite(total_risk):
                raise ValueError(f"total_risk must be finite, got {total_risk}")
            if not np.all(np.isfinite(contributions)):
                raise ValueError("All contributions must be finite")
        
        if abs(total_risk) < tolerance:
            return np.zeros_like(contributions)
        
        return contributions / total_risk
    
    @staticmethod
    def calculate_percent_factor_contributions(
        factor_contributions: np.ndarray,
        total_risk: float,
        validate: bool = True,
        tolerance: float = 1e-12
    ) -> np.ndarray:
        """
        Calculate factor percentage contributions normalized by total risk.
        
        Parameters
        ----------
        factor_contributions : np.ndarray
            Raw factor contributions (K,)
        total_risk : float
            Total risk for normalization
        validate : bool, default True
            Whether to validate inputs
        tolerance : float, default 1e-12
            Minimum total_risk to avoid division by zero
            
        Returns
        -------
        np.ndarray
            Percentage factor contributions (K,)
        """
        return RiskCalculator.calculate_percent_contributions(
            factor_contributions, total_risk, validate, tolerance
        )
    
    @staticmethod
    def calculate_normalized_contributions(
        contributions: np.ndarray,
        normalization_method: str = "total_risk",
        total_risk: Optional[float] = None,
        validate: bool = True,
        tolerance: float = 1e-12
    ) -> np.ndarray:
        """
        Calculate normalized contributions using various normalization methods.
        
        Parameters
        ----------
        contributions : np.ndarray
            Raw contributions (N,)
        normalization_method : str, default "total_risk"
            Normalization method: "total_risk", "absolute_sum", "sum_of_squares", "unit_sum"
        total_risk : float, optional
            Total risk for "total_risk" normalization. Required if using "total_risk" method.
        validate : bool, default True
            Whether to validate inputs
        tolerance : float, default 1e-12
            Minimum denominator to avoid division by zero
            
        Returns
        -------
        np.ndarray
            Normalized contributions (N,)
            
        Raises
        ------
        ValueError
            If normalization_method is invalid or required parameters are missing
        """
        contributions = np.asarray(contributions)
        
        if validate:
            if not np.all(np.isfinite(contributions)):
                raise ValueError("All contributions must be finite")
        
        if normalization_method == "total_risk":
            if total_risk is None:
                raise ValueError("total_risk must be provided for 'total_risk' normalization")
            return RiskCalculator.calculate_percent_contributions(
                contributions, total_risk, validate, tolerance
            )
        
        elif normalization_method == "absolute_sum":
            abs_sum = np.sum(np.abs(contributions))
            if abs_sum < tolerance:
                return np.zeros_like(contributions)
            return contributions / abs_sum
        
        elif normalization_method == "sum_of_squares":
            sum_sq = np.sum(contributions ** 2)
            if sum_sq < tolerance:
                return np.zeros_like(contributions)
            return contributions / np.sqrt(sum_sq)
        
        elif normalization_method == "unit_sum":
            total_sum = np.sum(contributions)
            if abs(total_sum) < tolerance:
                return np.zeros_like(contributions)
            return contributions / total_sum
        
        else:
            raise ValueError(
                f"Unknown normalization_method: {normalization_method}. "
                f"Valid methods: 'total_risk', 'absolute_sum', 'sum_of_squares', 'unit_sum'"
            )
    
    @staticmethod
    def calculate_asset_factor_contributions(
        beta: np.ndarray,
        factor_covar: np.ndarray,
        weights: np.ndarray,
        portfolio_volatility: float,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate asset-level contributions to factor risk.
        
        This method computes how much each individual asset contributes to the 
        portfolio's factor risk component. The calculation follows the formula:
        w_i * (B[i,:] @ Ω @ B^T @ w) / σ_p
        
        Parameters
        ----------
        beta : np.ndarray
            Factor loadings matrix (N x K)
        factor_covar : np.ndarray
            Factor covariance matrix (K x K)
        weights : np.ndarray
            Portfolio weights (N,)
        portfolio_volatility : float
            Pre-calculated portfolio volatility
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Asset-level factor contributions (raw, non-annualized)
        """
        if validate:
            if not np.isfinite(portfolio_volatility):
                raise ValueError(f"portfolio_volatility must be finite, got {portfolio_volatility}")
            if not np.all(np.isfinite(beta)):
                raise ValueError("All beta values must be finite")
            if not np.all(np.isfinite(factor_covar)):
                raise ValueError("All factor_covar values must be finite")
            if not np.all(np.isfinite(weights)):
                raise ValueError("All weights must be finite")
        
        weights = np.asarray(weights).flatten()
        
        if portfolio_volatility == 0:
            return np.zeros(weights.shape[0])
        
        # Factor contribution: w_i * (B[i,:] @ Ω @ B^T @ w) / σ_p
        factor_exposure = beta.T @ weights.reshape(-1, 1)  # K x 1
        factor_risk_contrib = factor_covar @ factor_exposure  # K x 1
        asset_factor_contrib = beta @ factor_risk_contrib  # N x 1
        
        return weights * asset_factor_contrib.flatten() / portfolio_volatility
    
    @staticmethod
    def calculate_asset_by_factor_contributions(
        beta: np.ndarray,
        factor_covar: np.ndarray,
        weights: np.ndarray,
        portfolio_volatility: float,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate asset x factor matrix of risk contributions.
        
        This is a detailed breakdown showing how each asset contributes
        to risk through each factor. Use with care as this is non-standard.
        
        Parameters
        ----------
        beta : np.ndarray
            Factor loadings matrix (N x K)
        factor_covar : np.ndarray
            Factor covariance matrix (K x K)
        weights : np.ndarray
            Portfolio weights (N,)
        portfolio_volatility : float
            Pre-calculated portfolio volatility
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Asset x factor contributions matrix (raw, non-annualized)
        """
        if validate:
            if not np.isfinite(portfolio_volatility):
                raise ValueError(f"portfolio_volatility must be finite, got {portfolio_volatility}")
            if not np.all(np.isfinite(beta)):
                raise ValueError("All beta values must be finite")
            if not np.all(np.isfinite(factor_covar)):
                raise ValueError("All factor_covar values must be finite")
            if not np.all(np.isfinite(weights)):
                raise ValueError("All weights must be finite")
        
        weights = np.asarray(weights).reshape(-1, 1)
        
        if portfolio_volatility == 0:
            return np.zeros((weights.shape[0], beta.shape[1]))
        
        # Asset by factor contributions: B * (F @ (B^T @ w) / vol).T
        factor_exposure = beta.T @ weights  # K x 1
        factor_risk_contrib = factor_covar @ factor_exposure  # K x 1
        contrib = beta * (factor_risk_contrib / portfolio_volatility).T  # N x K
        
        return contrib
    
    @staticmethod
    def calculate_weighted_betas(
        beta: np.ndarray,
        weights: np.ndarray,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate weighted betas matrix (Asset × Factor).
        
        For each asset i and factor j: weighted_beta[i,j] = beta[i,j] * weight[i]
        
        Parameters
        ----------
        beta : np.ndarray
            Factor loadings matrix (N x K)
        weights : np.ndarray
            Asset weights (N,)
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Weighted betas matrix (N x K)
        """
        if validate:
            if not np.all(np.isfinite(beta)):
                raise ValueError("All beta values must be finite")
            if not np.all(np.isfinite(weights)):
                raise ValueError("All weights must be finite")
            if beta.shape[0] != len(weights):
                raise ValueError(f"Beta rows ({beta.shape[0]}) must match weights length ({len(weights)})")
        
        weights = np.asarray(weights).reshape(-1, 1)  # Make it a column vector
        
        # Element-wise multiplication: each asset's beta weighted by its portfolio weight
        weighted_betas = beta * weights
        
        return weighted_betas
    
    @staticmethod
    def calculate_asset_allocation_factor_contributions(
        benchmark_beta: np.ndarray,
        benchmark_factor_covar: np.ndarray,
        active_weights: np.ndarray,
        active_risk: float,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate asset contributions to allocation factor risk.
        
        This method computes how much each asset contributes to the allocation
        component of factor risk in active risk decomposition.
        
        Parameters
        ----------
        benchmark_beta : np.ndarray
            Benchmark factor loadings matrix (N x K)
        benchmark_factor_covar : np.ndarray
            Benchmark factor covariance matrix (K x K)
        active_weights : np.ndarray
            Active weights (portfolio - benchmark) (N,)
        active_risk : float
            Total active risk (volatility)
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Asset allocation factor contributions (raw, non-annualized)
        """
        if validate:
            if not np.isfinite(active_risk):
                raise ValueError(f"active_risk must be finite, got {active_risk}")
            if not np.all(np.isfinite(benchmark_beta)):
                raise ValueError("All benchmark_beta values must be finite")
            if not np.all(np.isfinite(benchmark_factor_covar)):
                raise ValueError("All benchmark_factor_covar values must be finite")
            if not np.all(np.isfinite(active_weights)):
                raise ValueError("All active_weights must be finite")
        
        active_weights = np.asarray(active_weights).flatten()
        
        if active_risk == 0:
            return np.zeros(active_weights.shape[0])
        
        # Asset contributions to allocation factor risk
        factor_exposure = benchmark_beta.T @ active_weights.reshape(-1, 1)  # K x 1
        factor_risk_contrib = benchmark_factor_covar @ factor_exposure  # K x 1
        asset_factor_contrib = benchmark_beta @ factor_risk_contrib  # N x 1
        
        return active_weights * asset_factor_contrib.flatten() / active_risk
    
    @staticmethod
    def calculate_asset_specific_contributions(
        resvar: np.ndarray,
        weights: np.ndarray,
        portfolio_volatility: float,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate asset-level contributions to specific risk.
        
        This method computes how much each individual asset contributes to the 
        portfolio's specific (idiosyncratic) risk component.
        
        Parameters
        ----------
        resvar : np.ndarray
            Residual variance matrix (N x N) or vector (N,)
        weights : np.ndarray
            Portfolio weights (N,)
        portfolio_volatility : float
            Pre-calculated portfolio volatility
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Asset-level specific contributions (raw, non-annualized)
        """
        if validate:
            if not np.isfinite(portfolio_volatility):
                raise ValueError(f"portfolio_volatility must be finite, got {portfolio_volatility}")
            if not np.all(np.isfinite(resvar)):
                raise ValueError("All resvar values must be finite")
            if not np.all(np.isfinite(weights)):
                raise ValueError("All weights must be finite")
        
        weights = np.asarray(weights).flatten()
        
        if portfolio_volatility == 0:
            return np.zeros(weights.shape[0])
        
        # Handle both matrix and vector residual variance formats
        if resvar.ndim == 1:
            specific_contrib = resvar * weights
        else:
            specific_contrib = (resvar @ weights.reshape(-1, 1)).flatten()
        
        return weights * specific_contrib / portfolio_volatility
    
    @staticmethod
    def calculate_asset_allocation_specific_contributions(
        benchmark_resvar: np.ndarray,
        active_weights: np.ndarray,
        active_risk: float,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate asset contributions to allocation specific risk.
        
        This method computes how much each asset contributes to the allocation
        component of specific risk in active risk decomposition.
        
        Parameters
        ----------
        benchmark_resvar : np.ndarray
            Benchmark residual variance matrix (N x N) or vector (N,)
        active_weights : np.ndarray
            Active weights (portfolio - benchmark) (N,)
        active_risk : float
            Total active risk (volatility)
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Asset allocation specific contributions (raw, non-annualized)
        """
        if validate:
            if not np.isfinite(active_risk):
                raise ValueError(f"active_risk must be finite, got {active_risk}")
            if not np.all(np.isfinite(benchmark_resvar)):
                raise ValueError("All benchmark_resvar values must be finite")
            if not np.all(np.isfinite(active_weights)):
                raise ValueError("All active_weights must be finite")
        
        active_weights = np.asarray(active_weights).flatten()
        
        if active_risk == 0:
            return np.zeros(active_weights.shape[0])
        
        # Asset contributions to allocation specific risk
        if benchmark_resvar.ndim == 1:
            specific_contrib = benchmark_resvar * active_weights
        else:
            specific_contrib = (benchmark_resvar @ active_weights.reshape(-1, 1)).flatten()
        
        return active_weights * specific_contrib / active_risk
    
    @staticmethod
    def calculate_asset_selection_factor_contributions(
        portfolio_beta: np.ndarray,
        benchmark_beta: np.ndarray,
        active_factor_covar: np.ndarray,
        portfolio_weights: np.ndarray,
        active_risk: float,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate asset contributions to selection factor risk.
        
        This method computes how much each asset contributes to the selection
        component of factor risk in active risk decomposition.
        
        Parameters
        ----------
        portfolio_beta : np.ndarray
            Portfolio factor loadings matrix (N x K)
        benchmark_beta : np.ndarray
            Benchmark factor loadings matrix (N x K)
        active_factor_covar : np.ndarray
            Active factor covariance matrix (K x K)
        portfolio_weights : np.ndarray
            Portfolio weights (N,)
        active_risk : float
            Total active risk (volatility)
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Asset selection factor contributions (raw, non-annualized)
        """
        if validate:
            if not np.isfinite(active_risk):
                raise ValueError(f"active_risk must be finite, got {active_risk}")
            if not np.all(np.isfinite(portfolio_beta)):
                raise ValueError("All portfolio_beta values must be finite")
            if not np.all(np.isfinite(benchmark_beta)):
                raise ValueError("All benchmark_beta values must be finite")
            if not np.all(np.isfinite(active_factor_covar)):
                raise ValueError("All active_factor_covar values must be finite")
            if not np.all(np.isfinite(portfolio_weights)):
                raise ValueError("All portfolio_weights must be finite")
        
        portfolio_weights = np.asarray(portfolio_weights).flatten()
        
        if active_risk == 0:
            return np.zeros(portfolio_weights.shape[0])
        
        # Asset contributions to selection factor risk
        beta_diff = portfolio_beta - benchmark_beta
        selection_exposure = beta_diff.T @ portfolio_weights.reshape(-1, 1)  # K x 1
        factor_risk_contrib = active_factor_covar @ selection_exposure  # K x 1
        asset_factor_contrib = beta_diff @ factor_risk_contrib  # N x 1
        
        return portfolio_weights * asset_factor_contrib.flatten() / active_risk
    
    @staticmethod
    def calculate_asset_selection_specific_contributions(
        active_resvar: np.ndarray,
        portfolio_weights: np.ndarray,
        active_risk: float,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate asset contributions to selection specific risk.
        
        This method computes how much each asset contributes to the selection
        component of specific risk in active risk decomposition.
        
        Parameters
        ----------
        active_resvar : np.ndarray
            Active residual variance matrix (N x N) or vector (N,)
        portfolio_weights : np.ndarray
            Portfolio weights (N,)
        active_risk : float
            Total active risk (volatility)
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Asset selection specific contributions (raw, non-annualized)
        """
        if validate:
            if not np.isfinite(active_risk):
                raise ValueError(f"active_risk must be finite, got {active_risk}")
            if not np.all(np.isfinite(active_resvar)):
                raise ValueError("All active_resvar values must be finite")
            if not np.all(np.isfinite(portfolio_weights)):
                raise ValueError("All portfolio_weights must be finite")
        
        portfolio_weights = np.asarray(portfolio_weights).flatten()
        
        if active_risk == 0:
            return np.zeros(portfolio_weights.shape[0])
        
        # Asset contributions to selection specific risk
        if active_resvar.ndim == 1:
            specific_contrib = active_resvar * portfolio_weights
        else:
            specific_contrib = (active_resvar @ portfolio_weights.reshape(-1, 1)).flatten()
        
        return portfolio_weights * specific_contrib / active_risk
    
    @staticmethod
    def calculate_asset_cross_correlation_factor_contributions(
        cross_covar: np.ndarray,
        benchmark_beta: np.ndarray,
        active_beta: np.ndarray,
        active_weights: np.ndarray,
        portfolio_weights: np.ndarray,
        active_risk: float,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate asset contributions to cross-correlation factor risk.
        
        This method breaks down the cross-correlation term into factor components
        for each asset, enabling complete Euler identity decomposition.
        
        The decomposition follows:
        RC_cross_factor,i = (1/σ) * [d_i * (C * B_active * f_portfolio)_i + w_i * (C^T * B_benchmark * f_active)_i]
        
        where f represents factor exposures.
        
        Parameters
        ----------
        cross_covar : np.ndarray
            Cross-covariance matrix C (N x N)
        benchmark_beta : np.ndarray
            Benchmark factor loadings matrix (N x K)
        active_beta : np.ndarray
            Active factor loadings matrix (N x K)
        active_weights : np.ndarray
            Active weights d = portfolio - benchmark (N,)
        portfolio_weights : np.ndarray
            Portfolio weights w (N,)
        active_risk : float
            Total active risk (volatility)
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Asset cross-correlation factor contributions (raw, non-annualized)
        """
        if validate:
            if not np.isfinite(active_risk):
                raise ValueError(f"active_risk must be finite, got {active_risk}")
            if not np.all(np.isfinite(cross_covar)):
                raise ValueError("All cross_covar values must be finite")
            if not np.all(np.isfinite(benchmark_beta)):
                raise ValueError("All benchmark_beta values must be finite")
            if not np.all(np.isfinite(active_beta)):
                raise ValueError("All active_beta values must be finite")
            if not np.all(np.isfinite(active_weights)):
                raise ValueError("All active_weights must be finite")
            if not np.all(np.isfinite(portfolio_weights)):
                raise ValueError("All portfolio_weights must be finite")
        
        active_weights = np.asarray(active_weights).flatten()
        portfolio_weights = np.asarray(portfolio_weights).flatten()
        
        if active_risk == 0:
            return np.zeros(active_weights.shape[0])
        
        # Calculate factor exposures
        benchmark_factor_exp = benchmark_beta.T @ active_weights.reshape(-1, 1)  # K x 1
        active_factor_exp = active_beta.T @ portfolio_weights.reshape(-1, 1)  # K x 1
        
        # Factor-based cross-correlation terms
        # First term: d_i * (C * B_active * f_portfolio)_i
        cross_factor_portfolio = cross_covar @ active_beta @ active_factor_exp  # N x 1
        term1 = active_weights * cross_factor_portfolio.flatten()
        
        # Second term: w_i * (C^T * B_benchmark * f_active)_i  
        cross_factor_active = cross_covar.T @ benchmark_beta @ benchmark_factor_exp  # N x 1
        term2 = portfolio_weights * cross_factor_active.flatten()
        
        return (term1 + term2) / active_risk
    
    @staticmethod
    def calculate_factor_level_cross_contributions(
        cross_covar: np.ndarray,
        benchmark_beta: np.ndarray,
        active_beta: np.ndarray,
        active_weights: np.ndarray,
        portfolio_weights: np.ndarray,
        active_risk: float,
        validate: bool = True
    ) -> np.ndarray:
        """
        Calculate factor-level cross-correlation contributions.
        
        This method provides cross-correlation contributions at the factor level,
        enabling complete decomposition of the interaction term across factors.
        
        Parameters
        ----------
        cross_covar : np.ndarray
            Cross-covariance matrix C (N x N)
        benchmark_beta : np.ndarray
            Benchmark factor loadings matrix (N x K)
        active_beta : np.ndarray
            Active factor loadings matrix (N x K)
        active_weights : np.ndarray
            Active weights d = portfolio - benchmark (N,)
        portfolio_weights : np.ndarray
            Portfolio weights w (N,)
        active_risk : float
            Total active risk (volatility)
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        np.ndarray
            Factor-level cross-correlation contributions (K,)
        """
        if validate:
            if not np.isfinite(active_risk):
                raise ValueError(f"active_risk must be finite, got {active_risk}")
            if not np.all(np.isfinite(cross_covar)):
                raise ValueError("All cross_covar values must be finite")
            if not np.all(np.isfinite(benchmark_beta)):
                raise ValueError("All benchmark_beta values must be finite")
            if not np.all(np.isfinite(active_beta)):
                raise ValueError("All active_beta values must be finite")
            if not np.all(np.isfinite(active_weights)):
                raise ValueError("All active_weights must be finite")
            if not np.all(np.isfinite(portfolio_weights)):
                raise ValueError("All portfolio_weights must be finite")
        
        active_weights = np.asarray(active_weights).reshape(-1, 1)
        portfolio_weights = np.asarray(portfolio_weights).reshape(-1, 1)
        
        if active_risk == 0:
            return np.zeros(benchmark_beta.shape[1])
        
        # Factor-level cross-correlation: f_benchmark^T * (B_benchmark^T * C * B_active) * f_active / σ
        benchmark_factor_exp = benchmark_beta.T @ active_weights  # K x 1
        active_factor_exp = active_beta.T @ portfolio_weights  # K x 1
        
        # Cross-factor covariance matrix: B_benchmark^T * C * B_active (K x K)
        cross_factor_covar = benchmark_beta.T @ cross_covar @ active_beta
        
        # Factor contributions: 2 * f_benchmark * (cross_factor_covar * f_active) / σ
        factor_cross_contrib = 2.0 * benchmark_factor_exp.flatten() * (cross_factor_covar @ active_factor_exp).flatten()
        
        return factor_cross_contrib / active_risk
    
    @staticmethod
    def calculate_cross_correlation_marginal_contributions(
        cross_covar: np.ndarray,
        benchmark_beta: np.ndarray,
        benchmark_factor_covar: np.ndarray,
        active_beta: np.ndarray,
        active_factor_covar: np.ndarray,
        active_weights: np.ndarray,
        portfolio_weights: np.ndarray,
        active_risk: float,
        validate: bool = True
    ) -> dict:
        """
        Calculate comprehensive marginal contributions from cross-correlation term.
        
        This method provides detailed marginal analysis of how the cross-correlation
        term affects risk through different channels (allocation, selection, interaction).
        
        Parameters
        ----------
        cross_covar : np.ndarray
            Cross-covariance matrix C (N x N)
        benchmark_beta : np.ndarray
            Benchmark factor loadings matrix (N x K)
        benchmark_factor_covar : np.ndarray
            Benchmark factor covariance matrix (K x K)
        active_beta : np.ndarray
            Active factor loadings matrix (N x K)
        active_factor_covar : np.ndarray
            Active factor covariance matrix (K x K)
        active_weights : np.ndarray
            Active weights d = portfolio - benchmark (N,)
        portfolio_weights : np.ndarray
            Portfolio weights w (N,)
        active_risk : float
            Total active risk (volatility)
        validate : bool, default True
            Whether to validate inputs
            
        Returns
        -------
        dict
            Dictionary containing various marginal contribution breakdowns
        """
        if validate:
            if not np.isfinite(active_risk):
                raise ValueError(f"active_risk must be finite, got {active_risk}")
            # Additional validation as needed...
        
        active_weights = np.asarray(active_weights).flatten()
        portfolio_weights = np.asarray(portfolio_weights).flatten()
        
        if active_risk == 0:
            n_assets = len(active_weights)
            n_factors = benchmark_beta.shape[1]
            return {
                'asset_allocation_marginal': np.zeros(n_assets),
                'asset_selection_marginal': np.zeros(n_assets),
                'factor_allocation_marginal': np.zeros(n_factors),
                'factor_selection_marginal': np.zeros(n_factors),
                'total_cross_marginal': np.zeros(n_assets)
            }
        
        # Asset-level marginal contributions
        marginal_active, marginal_portfolio = RiskCalculator.calculate_marginal_cross_contributions(
            cross_covar, active_weights, portfolio_weights, active_risk, validate
        )
        
        # Factor-level marginal contributions
        benchmark_factor_exp = benchmark_beta.T @ active_weights.reshape(-1, 1)  # K x 1
        active_factor_exp = active_beta.T @ portfolio_weights.reshape(-1, 1)  # K x 1
        
        # Cross-factor marginal contributions
        cross_factor_covar = benchmark_beta.T @ cross_covar @ active_beta  # K x K
        
        factor_allocation_marginal = (cross_factor_covar @ active_factor_exp).flatten() / active_risk
        factor_selection_marginal = (cross_factor_covar.T @ benchmark_factor_exp).flatten() / active_risk
        
        return {
            'asset_allocation_marginal': marginal_active,
            'asset_selection_marginal': marginal_portfolio,
            'factor_allocation_marginal': factor_allocation_marginal,
            'factor_selection_marginal': factor_selection_marginal,
            'total_cross_marginal': marginal_active + marginal_portfolio
        }