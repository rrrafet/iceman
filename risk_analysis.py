"""
Simple risk analysis functions - 90% complexity reduction from the over-engineered system.

This module provides straightforward functions for portfolio risk analysis, replacing
the complex Strategy/Context/Decomposer/Schema architecture with simple, composable functions.

Key principles:
- Simple functions instead of class hierarchies
- Direct data access instead of complex schemas
- Mathematical accuracy preserved from RiskCalculator
- Easy to understand and maintain
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from .model import RiskModel
from .calculator import RiskCalculator
from .annualizer import RiskAnnualizer


def _create_weighted_betas_dict(
    weighted_betas_matrix: np.ndarray,
    asset_names: List[str], 
    factor_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Convert weighted betas matrix to nested dictionary format."""
    result = {}
    for i, asset in enumerate(asset_names):
        result[asset] = {}
        for j, factor in enumerate(factor_names):
            result[asset][factor] = float(weighted_betas_matrix[i, j])
    return result


def _create_asset_by_factor_dict(
    asset_by_factor_matrix: np.ndarray,
    asset_names: List[str],
    factor_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Convert asset by factor contributions matrix to nested dictionary format."""
    result = {}
    for i, asset in enumerate(asset_names):
        result[asset] = {}
        for j, factor in enumerate(factor_names):
            result[asset][factor] = float(asset_by_factor_matrix[i, j])
    return result


@dataclass
class RiskResult:
    """
    Simple container for risk analysis results.
    
    Replaces the over-engineered RiskResultSchema (1,866+ lines) with a clean,
    easy-to-use dataclass that contains ALL essential risk metrics from RiskCalculator.
    
    This dataclass preserves all mathematical results while eliminating complex abstractions.
    """
    
    # Core risk metrics  
    total_risk: float
    factor_risk_contribution: float  # Summable Euler contribution
    specific_risk_contribution: float  # Summable Euler contribution
    factor_volatility: float  # Standalone factor risk volatility
    specific_volatility: float  # Standalone specific risk volatility
    cross_correlation_volatility: float  # Standalone cross-correlation risk volatility

    # Attribution data
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    asset_contributions: Dict[str, float] = field(default_factory=dict)
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    
    # Detailed asset breakdowns (all calculator results)
    asset_factor_contributions: Dict[str, float] = field(default_factory=dict)
    asset_specific_contributions: Dict[str, float] = field(default_factory=dict)
    marginal_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Matrix data (for comprehensive analysis)
    weighted_betas: Optional[Dict[str, Dict[str, float]]] = None  # asset -> factor -> beta*weight
    asset_by_factor_contributions: Optional[Dict[str, Dict[str, float]]] = None  # asset -> factor -> contrib
    
    # Portfolio weights (for reference)
    portfolio_weights: Dict[str, float] = field(default_factory=dict)
    benchmark_weights: Optional[Dict[str, float]] = None
    active_weights: Optional[Dict[str, float]] = None
    
    # Advanced metrics (for active risk decomposition)
    allocation_factor_risk: Optional[float] = None
    allocation_specific_risk: Optional[float] = None  
    selection_factor_risk: Optional[float] = None
    selection_specific_risk: Optional[float] = None
    cross_correlation_risk_contribution: Optional[float] = 0.0
    cross_correlation_volatility: Optional[float] = 0.0
    
    # Active risk asset-level breakdowns
    asset_allocation_factor: Optional[Dict[str, float]] = None
    asset_allocation_specific: Optional[Dict[str, float]] = None
    asset_selection_factor: Optional[Dict[str, float]] = None
    asset_selection_specific: Optional[Dict[str, float]] = None
    asset_cross_correlation: Optional[Dict[str, float]] = None
    
    # Factor-level active decomposition
    factor_allocation_contributions: Optional[Dict[str, float]] = None
    factor_selection_contributions: Optional[Dict[str, float]] = None
    factor_cross_correlation: Optional[Dict[str, float]] = None
    
    # Metadata
    annualized: bool = False
    
    # Asset name mapping for visualization (component_id -> display_name)
    asset_name_mapping: Optional[Dict[str, str]] = None
    frequency: Optional[str] = None
    analysis_type: str = "portfolio"
    asset_names: List[str] = field(default_factory=list)
    factor_names: List[str] = field(default_factory=list)
    
    # Validation results
    validation_passed: bool = True
    validation_message: str = "OK"
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def factor_risk_pct(self) -> float:
        """Factor risk contribution as percentage of total risk."""
        return self.factor_risk_contribution / self.total_risk if self.total_risk > 0 else 0.0
    
    @property
    def specific_risk_pct(self) -> float:
        """Specific risk contribution as percentage of total risk."""
        return self.specific_risk_contribution / self.total_risk if self.total_risk > 0 else 0.0
    
    @property
    def total_risk_bps(self) -> float:
        """Total risk in basis points (for active risk)."""
        return self.total_risk * 10000
    
    @property
    def factor_risk_bps(self) -> float:
        """Factor risk contribution in basis points."""
        return self.factor_risk_contribution * 10000
    
    @property
    def specific_risk_bps(self) -> float:
        """Specific risk contribution in basis points."""
        return self.specific_risk_contribution * 10000
        
    def get_top_factor_contributions(self, n: int = 5) -> Dict[str, float]:
        """Get top N factor contributions by absolute value."""
        sorted_factors = sorted(
            self.factor_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return dict(sorted_factors[:n])
    
    def get_top_asset_contributions(self, n: int = 10) -> Dict[str, float]:
        """Get top N asset contributions by absolute value."""
        sorted_assets = sorted(
            self.asset_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return dict(sorted_assets[:n])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/export."""
        return {
            'total_risk': self.total_risk,
            'factor_risk_contribution': self.factor_risk_contribution,
            'specific_risk_contribution': self.specific_risk_contribution,
            'factor_volatility': self.factor_volatility,
            'specific_volatility': self.specific_volatility,
            'factor_risk_pct': self.factor_risk_pct,
            'specific_risk_pct': self.specific_risk_pct,
            'factor_contributions': self.factor_contributions,
            'asset_contributions': self.asset_contributions,
            'factor_exposures': self.factor_exposures,
            'analysis_type': self.analysis_type,
            'annualized': self.annualized,
            'validation_passed': self.validation_passed
        }


def analyze_portfolio_risk(
    model: RiskModel,
    weights: Union[np.ndarray, Dict[str, float]],
    asset_names: Optional[List[str]] = None,
    factor_names: Optional[List[str]] = None,
    asset_display_names: Optional[List[str]] = None,
    annualize: bool = True
) -> RiskResult:
    """
    Simple portfolio risk analysis function.
    
    Replaces the complex PortfolioAnalysisStrategy + SingleModelContext + RiskDecomposer
    with a single, easy-to-understand function.
    
    Parameters
    ----------
    model : RiskModel
        Risk model containing betas, factor covariance, residual variance
    weights : np.ndarray or Dict[str, float]
        Portfolio weights (must sum to ~1.0)
    asset_names : List[str], optional
        Asset names for labeling results (component IDs)
    factor_names : List[str], optional
        Factor names for labeling results
    asset_display_names : List[str], optional
        Display names for assets (for visualization)
    annualize : bool, default True
        Whether to annualize results based on model frequency
        
    Returns
    -------
    RiskResult
        Simple container with all essential risk metrics
    """
    
    # Convert weights to numpy array if needed
    if isinstance(weights, dict):
        if asset_names is None:
            asset_names = list(weights.keys())
        weights_array = np.array([weights[name] for name in asset_names])
    else:
        weights_array = np.asarray(weights).flatten()
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(len(weights_array))]
    
    # Set default factor names if not provided
    if factor_names is None:
        n_factors = model.beta.shape[1]
        factor_names = [f"factor_{i}" for i in range(n_factors)]
    
    # Core risk calculations using RiskCalculator
    portfolio_volatility = RiskCalculator.calculate_portfolio_volatility(model.covar, weights_array)
    
    # Factor analysis
    factor_exposures = RiskCalculator.calculate_factor_exposures(model.beta, weights_array)
    factor_contributions = RiskCalculator.calculate_factor_contributions(
        model.beta, model.factor_covar, weights_array, portfolio_volatility
    )
    factor_risk_contribution = np.sum(factor_contributions)
    factor_volatility = RiskCalculator.calculate_factor_risk(model.beta, model.factor_covar, weights_array)
    
    # Specific risk
    specific_contributions = RiskCalculator.calculate_specific_contributions(
        model.resvar, weights_array, portfolio_volatility
    )
    specific_risk_contribution = np.sum(specific_contributions)
    specific_volatility = RiskCalculator.calculate_specific_risk(model.resvar, weights_array)
    
    # Asset-level contributions  
    marginal_contributions = RiskCalculator.calculate_marginal_contributions(
        model.covar, weights_array, portfolio_volatility
    )
    asset_contributions = RiskCalculator.calculate_total_contributions(
        marginal_contributions, weights_array, portfolio_volatility
    )
    
    # Asset factor/specific breakdowns
    asset_factor_contributions = RiskCalculator.calculate_asset_factor_contributions(
        model.beta, model.factor_covar, weights_array, portfolio_volatility
    )
    asset_specific_contributions = RiskCalculator.calculate_asset_specific_contributions(
        model.resvar, weights_array, portfolio_volatility
    )
    
    # Create asset name mapping for visualization
    asset_name_mapping = None
    if asset_names and asset_display_names and len(asset_names) == len(asset_display_names):
        asset_name_mapping = dict(zip(asset_names, asset_display_names))
    
    # Validation
    validation = RiskCalculator.validate_risk_decomposition(
        portfolio_volatility,
        {'factor_risk': factor_risk_contribution, 'specific_risk': specific_risk_contribution}
    )
    
    # Apply annualization if requested
    if annualize and hasattr(model, 'frequency'):
        annualized_results = RiskAnnualizer.annualize_risk_results({
            'portfolio_volatility': portfolio_volatility,
            'factor_risk': factor_risk_contribution,
            'specific_risk': specific_risk_contribution
        }, model.frequency)
        
        annualized_factor_volatility = RiskAnnualizer.annualize_volatility(factor_volatility, model.frequency)
        annualized_specific_volatility = RiskAnnualizer.annualize_volatility(specific_volatility, model.frequency)
        
        # Annualize contributions
        annualized_factor_contrib = RiskAnnualizer.annualize_contributions(factor_contributions, model.frequency)
        annualized_asset_contrib = RiskAnnualizer.annualize_contributions(asset_contributions, model.frequency)
        annualized_asset_factor = RiskAnnualizer.annualize_contributions(asset_factor_contributions, model.frequency)
        annualized_asset_specific = RiskAnnualizer.annualize_contributions(asset_specific_contributions, model.frequency)
        
        # Create weighted betas matrix
        weighted_betas = _create_weighted_betas_dict(
            RiskCalculator.calculate_weighted_betas(model.beta, weights_array),
            asset_names, factor_names
        )
        
        # Create asset by factor contributions matrix
        asset_by_factor_matrix = RiskCalculator.calculate_asset_by_factor_contributions(
            model.beta, model.factor_covar, weights_array, portfolio_volatility
        )
        
        return RiskResult(
            total_risk=annualized_results['portfolio_volatility'],
            factor_risk_contribution=annualized_results['factor_risk'], 
            specific_risk_contribution=annualized_results['specific_risk'],
            factor_volatility=annualized_factor_volatility,
            specific_volatility=annualized_specific_volatility,
            factor_contributions=dict(zip(factor_names, annualized_factor_contrib)),
            asset_contributions=dict(zip(asset_names, annualized_asset_contrib)),
            factor_exposures=dict(zip(factor_names, factor_exposures)),
            asset_factor_contributions=dict(zip(asset_names, annualized_asset_factor)),
            asset_specific_contributions=dict(zip(asset_names, annualized_asset_specific)),
            marginal_contributions=dict(zip(asset_names, RiskAnnualizer.annualize_contributions(marginal_contributions, model.frequency))),
            weighted_betas=weighted_betas,
            asset_by_factor_contributions=_create_asset_by_factor_dict(
                RiskAnnualizer.annualize_contributions(asset_by_factor_matrix, model.frequency),
                asset_names, factor_names
            ),
            portfolio_weights=dict(zip(asset_names, weights_array)),
            asset_names=asset_names,
            factor_names=factor_names,
            asset_name_mapping=asset_name_mapping,
            annualized=True,
            frequency=model.frequency,
            validation_passed=validation['passes'],
            validation_message=validation['message'],
            validation_details=validation
        )
    else:
        # Create weighted betas matrix
        weighted_betas = _create_weighted_betas_dict(
            RiskCalculator.calculate_weighted_betas(model.beta, weights_array),
            asset_names, factor_names
        )
        
        # Create asset by factor contributions matrix
        asset_by_factor_matrix = RiskCalculator.calculate_asset_by_factor_contributions(
            model.beta, model.factor_covar, weights_array, portfolio_volatility
        )
        asset_by_factor_dict = _create_asset_by_factor_dict(
            asset_by_factor_matrix, asset_names, factor_names
        )
        
        return RiskResult(
            total_risk=portfolio_volatility,
            factor_risk_contribution=factor_risk_contribution,
            specific_risk_contribution=specific_risk_contribution,
            factor_volatility=factor_volatility,
            specific_volatility=specific_volatility,
            factor_contributions=dict(zip(factor_names, factor_contributions)),
            asset_contributions=dict(zip(asset_names, asset_contributions)),
            factor_exposures=dict(zip(factor_names, factor_exposures)),
            asset_factor_contributions=dict(zip(asset_names, asset_factor_contributions)),
            asset_specific_contributions=dict(zip(asset_names, asset_specific_contributions)),
            marginal_contributions=dict(zip(asset_names, marginal_contributions)),
            weighted_betas=weighted_betas,
            asset_by_factor_contributions=asset_by_factor_dict,
            portfolio_weights=dict(zip(asset_names, weights_array)),
            asset_names=asset_names,
            factor_names=factor_names,
            asset_name_mapping=asset_name_mapping,
            annualized=False,
            frequency=getattr(model, 'frequency', None),
            validation_passed=validation['passes'],
            validation_message=validation['message'],
            validation_details=validation
        )


def analyze_active_risk(
    portfolio_model: RiskModel,
    benchmark_model: RiskModel,
    portfolio_weights: Union[np.ndarray, Dict[str, float]],
    benchmark_weights: Union[np.ndarray, Dict[str, float]], 
    asset_names: Optional[List[str]] = None,
    factor_names: Optional[List[str]] = None,
    active_model: Optional[RiskModel] = None,
    cross_covar: Optional[np.ndarray] = None,
    asset_display_names: Optional[List[str]] = None,
    annualize: bool = True
) -> RiskResult:
    """
    Simple active risk analysis function.
    
    Replaces the complex ActiveRiskAnalysisStrategy + MultiModelContext + RiskDecomposer
    with a single, straightforward function.
    
    Parameters
    ----------
    portfolio_model : RiskModel
        Portfolio risk model
    benchmark_model : RiskModel  
        Benchmark risk model
    portfolio_weights : np.ndarray or Dict[str, float]
        Portfolio weights
    benchmark_weights : np.ndarray or Dict[str, float]
        Benchmark weights
    asset_names : List[str], optional
        Asset names for labeling
    factor_names : List[str], optional
        Factor names for labeling  
    active_model : RiskModel, optional
        Separate active risk model (if different from portfolio)
    cross_covar : np.ndarray, optional
        Cross-covariance matrix for correlation effects
    annualize : bool, default True
        Whether to annualize results
        
    Returns
    -------
    RiskResult
        Simple container with active risk decomposition
    """
    
    # Convert weights to arrays
    if isinstance(portfolio_weights, dict):
        if asset_names is None:
            asset_names = list(portfolio_weights.keys())
        port_weights_array = np.array([portfolio_weights[name] for name in asset_names])
    else:
        port_weights_array = np.asarray(portfolio_weights).flatten()
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(len(port_weights_array))]
    
    if isinstance(benchmark_weights, dict):
        bench_weights_array = np.array([benchmark_weights[name] for name in asset_names])
    else:
        bench_weights_array = np.asarray(benchmark_weights).flatten()
    
    active_weights_array = port_weights_array - bench_weights_array
    
    # Set factor names
    if factor_names is None:
        n_factors = portfolio_model.beta.shape[1]
        factor_names = [f"factor_{i}" for i in range(n_factors)]
    
    # Use active model if provided, otherwise use portfolio model
    if active_model is None:
        active_model = portfolio_model

    if cross_covar is None:
        cross_covar = RiskCalculator.calculate_cross_covariance(benchmark_model.asset_returns, active_model.asset_returns)
    # Calculate total active risk using RiskCalculator
    total_active_risk = RiskCalculator.calculate_active_risk(
        benchmark_model.covar,
        port_weights_array,
        bench_weights_array,
        active_model.covar,
        cross_covar
    )
    
    # Allocation components (benchmark model applied to active weights)
    allocation_factor_risk = RiskCalculator.calculate_factor_risk(
        benchmark_model.beta, benchmark_model.factor_covar, active_weights_array
    )
    allocation_specific_risk = RiskCalculator.calculate_specific_risk(
        benchmark_model.resvar, active_weights_array  
    )
    
    # Selection components (factor loading differences)
    beta_diff = portfolio_model.beta - benchmark_model.beta
    selection_factor_risk = RiskCalculator.calculate_factor_risk(
        beta_diff, active_model.factor_covar, port_weights_array
    )
    selection_specific_risk = RiskCalculator.calculate_specific_risk(
        active_model.resvar, port_weights_array
    )
    
    # Cross-correlation component
    cross_correlation_risk_contribution = 0.0
    cross_correlation_volatility = 0.0
    if cross_covar is not None:
        cross_euler = RiskCalculator.calculate_euler_cross_contributions(
            cross_covar, active_weights_array, port_weights_array, total_active_risk
        )
        cross_correlation_risk_contribution = cross_euler
        # For volatility, use the cross-variance contribution directly
        cross_variance = RiskCalculator.calculate_cross_variance_contribution(
            cross_covar, active_weights_array, port_weights_array
        )
        cross_correlation_volatility = np.sqrt(abs(cross_variance)) if cross_variance != 0 else 0.0
    
    # Calculate factor and specific risk volatilities (standalone)
    factor_volatility = np.sqrt(allocation_factor_risk ** 2 + selection_factor_risk ** 2)
    specific_volatility = np.sqrt(allocation_specific_risk ** 2 + selection_specific_risk ** 2)
    
    # Calculate factor and specific risk contributions (Euler contributions - summable)
    if total_active_risk > 0:
        allocation_factor_contribution = allocation_factor_risk ** 2 / total_active_risk
        allocation_specific_contribution = allocation_specific_risk ** 2 / total_active_risk
        selection_factor_contribution = selection_factor_risk ** 2 / total_active_risk  
        selection_specific_contribution = selection_specific_risk ** 2 / total_active_risk
        
        factor_risk_contribution = allocation_factor_contribution + selection_factor_contribution
        specific_risk_contribution = allocation_specific_contribution + selection_specific_contribution
    else:
        factor_risk_contribution = 0.0
        specific_risk_contribution = 0.0
    
    # Factor contributions (simplified approach)
    active_exposures = RiskCalculator.calculate_factor_exposures(
        portfolio_model.beta, port_weights_array
    ) - RiskCalculator.calculate_factor_exposures(
        benchmark_model.beta, bench_weights_array
    )
    
    # Factor-level active decomposition
    if total_active_risk > 0:
        # Allocation factor contributions
        allocation_exposures = RiskCalculator.calculate_factor_exposures(benchmark_model.beta, active_weights_array)
        allocation_marginal = (benchmark_model.factor_covar @ allocation_exposures) / total_active_risk
        factor_allocation_contributions = allocation_exposures * allocation_marginal
        
        # Selection factor contributions  
        beta_diff = portfolio_model.beta - benchmark_model.beta
        selection_exposures = RiskCalculator.calculate_factor_exposures(beta_diff, port_weights_array)
        selection_marginal = (active_model.factor_covar @ selection_exposures) / total_active_risk
        factor_selection_contributions = selection_exposures * selection_marginal
        
        # Cross-correlation factor contributions
        factor_cross_contributions = np.zeros(len(factor_names))
        if cross_covar is not None:
            try:
                factor_cross_contributions = RiskCalculator.calculate_factor_level_cross_contributions(
                    cross_covar, benchmark_model.beta, active_model.beta,
                    active_weights_array, port_weights_array, total_active_risk
                )
            except:
                pass  # Fallback to zeros if calculation fails
        
        # Total factor contributions
        factor_contributions = factor_allocation_contributions + factor_selection_contributions  # factor_cross_contributions
    else:
        factor_allocation_contributions = np.zeros(len(factor_names))
        factor_selection_contributions = np.zeros(len(factor_names))
        factor_cross_contributions = np.zeros(len(factor_names))
        factor_contributions = np.zeros_like(active_exposures)
    
    # Detailed asset-level active risk contributions using RiskCalculator
    if total_active_risk > 0:
        # Asset allocation factor contributions
        asset_allocation_factor = RiskCalculator.calculate_asset_allocation_factor_contributions(
            benchmark_model.beta, benchmark_model.factor_covar, active_weights_array, total_active_risk
        )
        
        # Asset allocation specific contributions  
        asset_allocation_specific = RiskCalculator.calculate_asset_allocation_specific_contributions(
            benchmark_model.resvar, active_weights_array, total_active_risk
        )
        
        # Asset selection factor contributions
        asset_selection_factor = RiskCalculator.calculate_asset_selection_factor_contributions(
            portfolio_model.beta, benchmark_model.beta, active_model.factor_covar,
            port_weights_array, total_active_risk
        )
        
        # Asset selection specific contributions
        asset_selection_specific = RiskCalculator.calculate_asset_selection_specific_contributions(
            active_model.resvar, port_weights_array, total_active_risk
        )
        
        # Asset cross-correlation contributions
        asset_cross_correlation = np.zeros_like(active_weights_array)
        if cross_covar is not None:
            asset_cross_correlation = RiskCalculator.calculate_asset_level_cross_contributions(
                cross_covar, active_weights_array, port_weights_array, total_active_risk
            )
        
        # Total asset contributions
        asset_contributions = (asset_allocation_factor + asset_allocation_specific + 
                             asset_selection_factor + asset_selection_specific + asset_cross_correlation)
    else:
        asset_allocation_factor = np.zeros_like(active_weights_array)
        asset_allocation_specific = np.zeros_like(active_weights_array)
        asset_selection_factor = np.zeros_like(active_weights_array)
        asset_selection_specific = np.zeros_like(active_weights_array)
        asset_cross_correlation = np.zeros_like(active_weights_array)
        asset_contributions = np.zeros_like(active_weights_array)
    
    # Asset-level marginal contributions (MISSING from active risk analysis)
    if total_active_risk > 0:
        marginal_contributions = RiskCalculator.calculate_marginal_contributions(
            active_model.covar, active_weights_array, total_active_risk
        )
    else:
        marginal_contributions = np.zeros_like(active_weights_array)
    
    # Create asset name mapping for visualization
    asset_name_mapping = None
    if asset_names and asset_display_names and len(asset_names) == len(asset_display_names):
        asset_name_mapping = dict(zip(asset_names, asset_display_names))
    
    # Create matrix data for visualization (using active weights)
    weighted_betas = _create_weighted_betas_dict(
        RiskCalculator.calculate_weighted_betas(active_model.beta, active_weights_array),
        asset_names, factor_names
    )
    
    # Create asset by factor contributions matrix (for active risk)
    # Decompose into allocation and selection components for mathematical consistency
    
    # Allocation component: benchmark exposures with active weights
    allocation_asset_by_factor = RiskCalculator.calculate_asset_by_factor_contributions(
        beta=benchmark_model.beta,
        factor_covar=benchmark_model.factor_covar, 
        weights=active_weights_array,  # Active weights (portfolio - benchmark)
        portfolio_volatility=total_active_risk
    )
    
    # Selection component: beta differences with portfolio weights
    beta_diff = portfolio_model.beta - benchmark_model.beta
    selection_asset_by_factor = RiskCalculator.calculate_asset_by_factor_contributions(
        beta=beta_diff,
        factor_covar=active_model.factor_covar,
        weights=port_weights_array,  # Portfolio weights
        portfolio_volatility=total_active_risk
    )
    
    # Total asset by factor contributions = allocation + selection
    asset_by_factor_matrix = allocation_asset_by_factor + selection_asset_by_factor

    # Validation
    validation = RiskCalculator.validate_active_risk_euler_identity(
        total_active_risk,
        allocation_factor_risk,
        allocation_specific_risk, 
        selection_factor_risk,
        selection_specific_risk,
        cross_correlation_risk_contribution
    )
    
    # Apply annualization if requested
    frequency = getattr(portfolio_model, 'frequency', 'D')
    if annualize:
        annualized_results = RiskAnnualizer.annualize_risk_results({
            'portfolio_volatility': total_active_risk,
            'factor_risk_contribution': factor_risk_contribution,
            'specific_risk_contribution': specific_risk_contribution,
            'cross_correlation_risk_contribution': cross_correlation_risk_contribution
        }, frequency)
        
        annualized_factor_volatility = RiskAnnualizer.annualize_volatility(factor_volatility, frequency)
        annualized_specific_volatility = RiskAnnualizer.annualize_volatility(specific_volatility, frequency)
        
        annualized_factor_contrib = RiskAnnualizer.annualize_contributions(factor_contributions, frequency)
        annualized_asset_contrib = RiskAnnualizer.annualize_contributions(asset_contributions, frequency)
        
        # Annualize all detailed contributions
        annualized_asset_allocation_factor = RiskAnnualizer.annualize_contributions(asset_allocation_factor, frequency)
        annualized_asset_allocation_specific = RiskAnnualizer.annualize_contributions(asset_allocation_specific, frequency)
        annualized_asset_selection_factor = RiskAnnualizer.annualize_contributions(asset_selection_factor, frequency)
        annualized_asset_selection_specific = RiskAnnualizer.annualize_contributions(asset_selection_specific, frequency)
        annualized_asset_cross = RiskAnnualizer.annualize_contributions(asset_cross_correlation, frequency)
        
        annualized_factor_allocation = RiskAnnualizer.annualize_contributions(factor_allocation_contributions, frequency)
        annualized_factor_selection = RiskAnnualizer.annualize_contributions(factor_selection_contributions, frequency)
        annualized_factor_cross = RiskAnnualizer.annualize_contributions(factor_cross_contributions, frequency)
        
        return RiskResult(
            total_risk=annualized_results['portfolio_volatility'],
            factor_risk_contribution=annualized_results['factor_risk_contribution'],
            specific_risk_contribution=annualized_results['specific_risk_contribution'],
            factor_volatility=annualized_factor_volatility,
            specific_volatility=annualized_specific_volatility,
            factor_contributions=dict(zip(factor_names, annualized_factor_contrib)),
            asset_contributions=dict(zip(asset_names, annualized_asset_contrib)),
            factor_exposures=dict(zip(factor_names, active_exposures)),
            marginal_contributions=dict(zip(asset_names, RiskAnnualizer.annualize_contributions(marginal_contributions, frequency))),
            
            # Portfolio weights
            portfolio_weights=dict(zip(asset_names, port_weights_array)),
            benchmark_weights=dict(zip(asset_names, bench_weights_array)),
            active_weights=dict(zip(asset_names, active_weights_array)),
            
            # Active risk decomposition
            allocation_factor_risk=RiskAnnualizer.annualize_volatility(allocation_factor_risk, frequency),
            allocation_specific_risk=RiskAnnualizer.annualize_volatility(allocation_specific_risk, frequency),
            selection_factor_risk=RiskAnnualizer.annualize_volatility(selection_factor_risk, frequency),
            selection_specific_risk=RiskAnnualizer.annualize_volatility(selection_specific_risk, frequency),
            cross_correlation_risk_contribution=RiskAnnualizer.annualize_contributions(np.array([cross_correlation_risk_contribution]), frequency)[0],
            cross_correlation_volatility=RiskAnnualizer.annualize_volatility(cross_correlation_volatility, frequency),
            
            # Asset-level active breakdowns
            asset_allocation_factor=dict(zip(asset_names, annualized_asset_allocation_factor)),
            asset_allocation_specific=dict(zip(asset_names, annualized_asset_allocation_specific)),
            asset_selection_factor=dict(zip(asset_names, annualized_asset_selection_factor)),
            asset_selection_specific=dict(zip(asset_names, annualized_asset_selection_specific)),
            asset_cross_correlation=dict(zip(asset_names, annualized_asset_cross)),
            
            # Factor-level active breakdowns
            factor_allocation_contributions=dict(zip(factor_names, annualized_factor_allocation)),
            factor_selection_contributions=dict(zip(factor_names, annualized_factor_selection)),
            factor_cross_correlation=dict(zip(factor_names, annualized_factor_cross)),
            
            # Matrix data for heatmaps
            weighted_betas=weighted_betas,
            asset_by_factor_contributions = _create_asset_by_factor_dict(
                RiskAnnualizer.annualize_contributions(asset_by_factor_matrix), asset_names, factor_names
            ),

            # Metadata
            asset_names=asset_names,
            factor_names=factor_names,
            asset_name_mapping=asset_name_mapping,
            analysis_type="active",
            annualized=True,
            frequency=frequency,
            validation_passed=validation['euler_identity_passes'],
            validation_message=validation['message'],
            validation_details=validation
        )
    else:
        return RiskResult(
            total_risk=total_active_risk,
            factor_risk_contribution=factor_risk_contribution,
            specific_risk_contribution=specific_risk_contribution,
            factor_volatility=factor_volatility,
            specific_volatility=specific_volatility,
            factor_contributions=dict(zip(factor_names, factor_contributions)),
            asset_contributions=dict(zip(asset_names, asset_contributions)),
            factor_exposures=dict(zip(factor_names, active_exposures)),
            marginal_contributions=dict(zip(asset_names, marginal_contributions)),
            
            # Portfolio weights
            portfolio_weights=dict(zip(asset_names, port_weights_array)),
            benchmark_weights=dict(zip(asset_names, bench_weights_array)),
            active_weights=dict(zip(asset_names, active_weights_array)),
            
            # Active risk decomposition
            allocation_factor_risk=allocation_factor_risk,
            allocation_specific_risk=allocation_specific_risk,
            selection_factor_risk=selection_factor_risk, 
            selection_specific_risk=selection_specific_risk,
            cross_correlation_risk_contribution=cross_correlation_risk_contribution,
            cross_correlation_volatility=cross_correlation_volatility,
            
            # Asset-level active breakdowns
            asset_allocation_factor=dict(zip(asset_names, asset_allocation_factor)),
            asset_allocation_specific=dict(zip(asset_names, asset_allocation_specific)),
            asset_selection_factor=dict(zip(asset_names, asset_selection_factor)),
            asset_selection_specific=dict(zip(asset_names, asset_selection_specific)),
            asset_cross_correlation=dict(zip(asset_names, asset_cross_correlation)),
            
            # Factor-level active breakdowns
            factor_allocation_contributions=dict(zip(factor_names, factor_allocation_contributions)),
            factor_selection_contributions=dict(zip(factor_names, factor_selection_contributions)),
            factor_cross_correlation=dict(zip(factor_names, factor_cross_contributions)),
            
            # Matrix data for heatmaps
            weighted_betas=weighted_betas,
            asset_by_factor_contributions = _create_asset_by_factor_dict(
                asset_by_factor_matrix, asset_names, factor_names
            ),
            # Metadata
            asset_names=asset_names,
            factor_names=factor_names,
            asset_name_mapping=asset_name_mapping,
            analysis_type="active",
            annualized=False,
            frequency=frequency,
            validation_passed=validation['euler_identity_passes'],
            validation_message=validation['message'],
            validation_details=validation
        )


def decompose_portfolio_risk(
    portfolio_graph,
    component_id: str,
    factor_returns: Optional[np.ndarray] = None,
    annualize: bool = True
) -> RiskResult:
    """
    Integration function for portfolio graph traversal.
    
    Extracts model and weights from portfolio graph and performs risk analysis.
    This preserves the visitor pattern integration while using the simplified API.
    
    Parameters
    ----------
    portfolio_graph : PortfolioGraph
        Portfolio hierarchy graph
    component_id : str
        Component to analyze
    factor_returns : np.ndarray, optional
        Factor returns for model estimation (if needed)
    annualize : bool, default True
        Whether to annualize results
        
    Returns
    -------
    RiskResult
        Risk analysis results for the component
    """
    
    # Extract risk model and weights from the portfolio graph
    # This assumes the FactorRiskDecompositionVisitor has already populated
    # the metric store with the necessary risk models and weights
    
    metric_store = portfolio_graph.metric_store
    if not metric_store:
        raise ValueError("Portfolio graph must have a metric store with risk models")
    
    # Try to get portfolio model from metric store
    portfolio_model_metric = metric_store.get_metric(component_id, 'portfolio_model')
    if not portfolio_model_metric:
        raise ValueError(f"No portfolio risk model found for component {component_id}")
    
    portfolio_model = portfolio_model_metric.value()
    
    # Try to get portfolio weights
    portfolio_weight_metric = metric_store.get_metric(component_id, 'portfolio_weight')  
    if not portfolio_weight_metric:
        raise ValueError(f"No portfolio weights found for component {component_id}")
    
    portfolio_weights = portfolio_weight_metric.value()
    
    # Check if benchmark model exists for active risk analysis
    benchmark_model_metric = metric_store.get_metric(component_id, 'benchmark_model')
    benchmark_weight_metric = metric_store.get_metric(component_id, 'benchmark_weight')
    
    if benchmark_model_metric and benchmark_weight_metric:
        # Perform active risk analysis
        benchmark_model = benchmark_model_metric.value()
        benchmark_weights = benchmark_weight_metric.value()
        
        # Try to get active model if available
        active_model_metric = metric_store.get_metric(component_id, 'active_model')
        active_model = active_model_metric.value() if active_model_metric else None
        
        return analyze_active_risk(
            portfolio_model, benchmark_model,
            portfolio_weights, benchmark_weights,
            active_model=active_model,
            annualize=annualize
        )
    else:
        # Perform simple portfolio risk analysis
        return analyze_portfolio_risk(
            portfolio_model, 
            portfolio_weights,
            annualize=annualize
        )