"""
Risk analysis context classes for encapsulating different risk decomposition scenarios.

This module provides context classes that encapsulate all the data and models needed
for different types of risk analysis, promoting type safety and clean interfaces.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from .model import RiskModel


class RiskContext(ABC):
    """
    Abstract base class for risk analysis contexts.
    
    A context encapsulates all the data needed for a specific type of risk analysis,
    including risk models, weights, and configuration parameters.
    """
    
    def __init__(self, annualize: bool = True):
        self.annualize = annualize
    
    @property
    @abstractmethod
    def frequency(self) -> str:
        """Data frequency for annualization"""
        pass
    
    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """Validate the context data for consistency"""
        pass
    
    @abstractmethod
    def get_asset_names(self) -> List[str]:
        """Get list of asset names"""
        pass
    
    @abstractmethod
    def get_factor_names(self) -> List[str]:
        """Get list of factor names"""
        pass


class SingleModelContext(RiskContext):
    """
    Context for single-model risk analysis (traditional portfolio risk decomposition).
    
    This context encapsulates a single risk model and portfolio weights,
    suitable for standard portfolio risk analysis.
    """
    
    def __init__(
        self,
        model: RiskModel,
        weights: np.ndarray,
        annualize: bool = True
    ):
        """
        Initialize single-model context.
        
        Parameters
        ----------
        model : RiskModel
            Risk model containing factor loadings, covariances, etc.
        weights : np.ndarray
            Portfolio weights
        annualize : bool, default True
            Whether to annualize risk metrics
        """
        super().__init__(annualize)
        self.model = model
        self.weights = np.asarray(weights).flatten()
        
        # Validate dimensions
        validation = self.validate()
        if not validation['passes']:
            raise ValueError(f"Context validation failed: {validation['message']}")
    
    @property
    def frequency(self) -> str:
        return self.model.frequency
    
    def validate(self) -> Dict[str, Any]:
        """Validate model and weights compatibility"""
        issues = []
        
        # Check weight dimensions
        if self.weights.shape[0] != self.model.beta.shape[0]:
            issues.append(f"Weight dimension {self.weights.shape[0]} != model asset dimension {self.model.beta.shape[0]}")
        
        # Check weight sum (should be close to 1.0 for meaningful interpretation)
        weight_sum = np.sum(self.weights)
        if abs(weight_sum - 1.0) > 0.01:  # 1% tolerance
            issues.append(f"Weights sum to {weight_sum:.4f}, expected ≈ 1.0")
        
        # Check for NaN/inf values
        if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
            issues.append("Weights contain NaN or infinite values")
        
        # Check model matrices
        if np.any(np.isnan(self.model.beta)):
            issues.append("Model beta contains NaN values")
        
        if np.any(np.isnan(self.model.factor_covar)):
            issues.append("Model factor covariance contains NaN values")
        
        return {
            'passes': len(issues) == 0,
            'issues': issues,
            'message': '; '.join(issues) if issues else 'Validation passed'
        }
    
    def get_asset_names(self) -> List[str]:
        """Get asset names from model"""
        if hasattr(self.model, 'symbols'):
            return list(self.model.symbols)
        else:
            # Fallback: generate generic names based on weight dimensions
            n_assets = len(self.weights)
            return [f"Asset_{i+1}" for i in range(n_assets)]
    
    def get_factor_names(self) -> List[str]:
        """Get factor names from model"""
        if hasattr(self.model, 'factor_names'):
            return list(self.model.factor_names)
        else:
            # Fallback: generate generic names based on beta dimensions
            n_factors = self.model.beta.shape[1]
            return [f"Factor_{i+1}" for i in range(n_factors)]


class MultiModelContext(RiskContext):
    """
    Context for multi-model risk analysis (active risk decomposition).
    
    This context encapsulates separate risk models for portfolio, benchmark,
    and optionally active returns, along with their respective weights.
    """
    
    def __init__(
        self,
        portfolio_model: RiskModel,
        benchmark_model: RiskModel,
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray,
        active_model: Optional[RiskModel] = None,
        cross_covar: Optional[np.ndarray] = None,
        annualize: bool = True
    ):
        """
        Initialize multi-model context.
        
        Parameters
        ----------
        portfolio_model : RiskModel
            Risk model for portfolio
        benchmark_model : RiskModel
            Risk model for benchmark
        portfolio_weights : np.ndarray
            Portfolio weights
        benchmark_weights : np.ndarray
            Benchmark weights
        active_model : RiskModel, optional
            Risk model for active returns. If None, uses portfolio_model.
        cross_covar : np.ndarray, optional
            Cross-covariance matrix between benchmark and active returns (N x N).
            If None, cross-correlation term is assumed to be zero.
        annualize : bool, default True
            Whether to annualize risk metrics
        """
        super().__init__(annualize)
        self.portfolio_model = portfolio_model
        self.benchmark_model = benchmark_model
        self.active_model = active_model or portfolio_model
        self.portfolio_weights = np.asarray(portfolio_weights).flatten()
        self.benchmark_weights = np.asarray(benchmark_weights).flatten()
        self.cross_covar = cross_covar
        
        # Validate compatibility
        validation = self.validate()
        if not validation['passes']:
            raise ValueError(f"Context validation failed: {validation['message']}")
    
    @property
    def frequency(self) -> str:
        return self.portfolio_model.frequency
    
    @property
    def active_weights(self) -> np.ndarray:
        """Active weights (portfolio - benchmark)"""
        return self.portfolio_weights - self.benchmark_weights
    
    def validate(self) -> Dict[str, Any]:
        """Validate model compatibility and weights"""
        issues = []
        
        # Check weight dimensions match each other
        if self.portfolio_weights.shape != self.benchmark_weights.shape:
            issues.append(f"Portfolio and benchmark weight shapes don't match: "
                         f"{self.portfolio_weights.shape} vs {self.benchmark_weights.shape}")
        
        # Check weight dimensions match models
        if self.portfolio_weights.shape[0] != self.portfolio_model.beta.shape[0]:
            issues.append(f"Portfolio weight dimension {self.portfolio_weights.shape[0]} != "
                         f"portfolio model asset dimension {self.portfolio_model.beta.shape[0]}")
        
        if self.benchmark_weights.shape[0] != self.benchmark_model.beta.shape[0]:
            issues.append(f"Benchmark weight dimension {self.benchmark_weights.shape[0]} != "
                         f"benchmark model asset dimension {self.benchmark_model.beta.shape[0]}")
        
        # Check model compatibility
        models_to_check = [
            ("portfolio", "benchmark", self.portfolio_model, self.benchmark_model),
            ("portfolio", "active", self.portfolio_model, self.active_model),
            ("benchmark", "active", self.benchmark_model, self.active_model)
        ]
        
        for name1, name2, model1, model2 in models_to_check:
            if model1.beta.shape != model2.beta.shape:
                issues.append(f"{name1.title()} and {name2} model beta shapes don't match: "
                             f"{model1.beta.shape} vs {model2.beta.shape}")
            
            if model1.factor_covar.shape != model2.factor_covar.shape:
                issues.append(f"{name1.title()} and {name2} factor covariance shapes don't match")
            
            if model1.frequency != model2.frequency:
                issues.append(f"{name1.title()} and {name2} frequencies don't match: "
                             f"{model1.frequency} vs {model2.frequency}")
        
        # Check weight sums
        port_sum = np.sum(self.portfolio_weights)
        bench_sum = np.sum(self.benchmark_weights)
        active_sum = np.sum(self.active_weights)
        
        if abs(port_sum - 1.0) > 0.01:
            issues.append(f"Portfolio weights sum to {port_sum:.4f}, expected ≈ 1.0")
        
        if abs(bench_sum - 1.0) > 0.01:
            issues.append(f"Benchmark weights sum to {bench_sum:.4f}, expected ≈ 1.0")
        
        if abs(active_sum) > 0.01:  # Active weights should sum to ~0
            issues.append(f"Active weights sum to {active_sum:.4f}, expected ≈ 0.0")
        
        # Check for NaN/inf values
        for name, weights in [("Portfolio", self.portfolio_weights), ("Benchmark", self.benchmark_weights)]:
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                issues.append(f"{name} weights contain NaN or infinite values")
        
        # Validate cross-covariance matrix if provided
        if self.cross_covar is not None:
            cross_covar = np.asarray(self.cross_covar)
            
            # Check dimensions
            expected_shape = (self.portfolio_weights.shape[0], self.portfolio_weights.shape[0])
            if cross_covar.shape != expected_shape:
                issues.append(f"Cross-covariance matrix shape {cross_covar.shape} != expected {expected_shape}")
            
            # Check for finite values
            if not np.all(np.isfinite(cross_covar)):
                issues.append("Cross-covariance matrix contains NaN or infinite values")
            
            # Check if matrix is square
            if cross_covar.shape[0] != cross_covar.shape[1]:
                issues.append(f"Cross-covariance matrix must be square, got shape {cross_covar.shape}")
        
        return {
            'passes': len(issues) == 0,
            'issues': issues,
            'message': '; '.join(issues) if issues else 'Validation passed'
        }
    
    def get_asset_names(self) -> List[str]:
        """Get asset names from portfolio model"""
        if hasattr(self.portfolio_model, 'symbols'):
            return list(self.portfolio_model.symbols)
        else:
            # Fallback: generate generic names based on weight dimensions
            n_assets = len(self.portfolio_weights)
            return [f"Asset_{i+1}" for i in range(n_assets)]
    
    def get_factor_names(self) -> List[str]:
        """Get factor names from portfolio model"""
        if hasattr(self.portfolio_model, 'factor_names'):
            return list(self.portfolio_model.factor_names)
        else:
            # Fallback: generate generic names based on beta dimensions
            n_factors = self.portfolio_model.beta.shape[1]
            return [f"Factor_{i+1}" for i in range(n_factors)]


class CustomContext(RiskContext):
    """
    Flexible context for custom risk analysis scenarios.
    
    This context allows for arbitrary combinations of models and weights,
    suitable for advanced or experimental risk decomposition approaches.
    """
    
    def __init__(
        self,
        models: Dict[str, RiskModel],
        weights: Dict[str, np.ndarray],
        frequency: str,
        asset_names: List[str],
        factor_names: List[str],
        annualize: bool = True
    ):
        """
        Initialize custom context.
        
        Parameters
        ----------
        models : Dict[str, RiskModel]
            Dictionary of named risk models
        weights : Dict[str, np.ndarray]
            Dictionary of named weight arrays
        frequency : str
            Data frequency for annualization
        asset_names : List[str]
            List of asset names
        factor_names : List[str]
            List of factor names
        annualize : bool, default True
            Whether to annualize risk metrics
        """
        super().__init__(annualize)
        self.models = models
        self.weights = {name: np.asarray(w).flatten() for name, w in weights.items()}
        self._frequency = frequency
        self.asset_names = asset_names
        self.factor_names = factor_names
        
        # Validate
        validation = self.validate()
        if not validation['passes']:
            raise ValueError(f"Context validation failed: {validation['message']}")
    
    @property
    def frequency(self) -> str:
        return self._frequency
    
    def validate(self) -> Dict[str, Any]:
        """Validate custom context consistency"""
        issues = []
        
        # Check that models and weights have consistent keys
        if set(self.models.keys()) != set(self.weights.keys()):
            issues.append("Model and weight keys don't match")
        
        # Check each model-weight pair
        for name in self.models.keys():
            if name in self.weights:
                model = self.models[name]
                weights = self.weights[name]
                
                if weights.shape[0] != model.beta.shape[0]:
                    issues.append(f"{name}: weight dimension {weights.shape[0]} != "
                                 f"model asset dimension {model.beta.shape[0]}")
                
                if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                    issues.append(f"{name}: weights contain NaN or infinite values")
        
        return {
            'passes': len(issues) == 0,
            'issues': issues,
            'message': '; '.join(issues) if issues else 'Validation passed'
        }
    
    def get_asset_names(self) -> List[str]:
        """Get asset names"""
        return self.asset_names.copy()
    
    def get_factor_names(self) -> List[str]:
        """Get factor names"""
        return self.factor_names.copy()
    
    def get_model(self, name: str) -> Optional[RiskModel]:
        """Get a specific model by name"""
        return self.models.get(name)
    
    def get_weights(self, name: str) -> Optional[np.ndarray]:
        """Get specific weights by name"""
        return self.weights.get(name)


class HierarchicalModelContext(RiskContext):
    """
    Context for hierarchical risk model analysis from portfolio tree nodes.
    
    This context encapsulates unified matrices from portfolio hierarchy analysis,
    creating comprehensive risk models for portfolio, benchmark, and active analysis
    at any node level in the portfolio tree.
    
    Key features:
    - Manages multiple related risk models from hierarchical data
    - Provides unified access to portfolio tree risk analysis
    - Creates decomposer instances for comprehensive risk attribution
    - Handles aggregated descendant leaf data with normalized weights
    """
    
    def __init__(
        self, 
        unified_matrices: Dict[str, Dict[str, Any]], 
        annualize: bool = True
    ):
        """
        Initialize hierarchical model context from unified matrices.
        
        Parameters
        ----------
        unified_matrices : Dict[str, Dict[str, Any]]
            Unified matrices containing 'portfolio', 'benchmark', and 'active' risk data
        annualize : bool, default True
            Whether to annualize risk metrics
        """
        super().__init__(annualize)
        self.unified_matrices = unified_matrices
        
        # Extract factor names from factor_returns DataFrame
        self.factor_names = self._extract_factor_names()
        self.descendant_leaves = unified_matrices['portfolio']['descendant_leaves']
        
        # Validate the unified matrices
        validation = self.validate()
        if not validation['passes']:
            raise ValueError(f"Context validation failed: {validation['message']}")
        
        # Create individual risk models for each type
        self._create_risk_models()
        
        # Create decomposers for each risk type using new architecture
        self._create_decomposers()
        
        # Store comprehensive results
        self._store_results()
    
    @property
    def frequency(self) -> str:
        """Get frequency from portfolio model data"""
        return self.unified_matrices['portfolio']['frequency']
    
    def validate(self) -> Dict[str, Any]:
        """Validate unified matrices consistency"""
        issues = []
        
        # Check that all required matrix types exist
        required_types = ['portfolio', 'benchmark', 'active']
        for matrix_type in required_types:
            if matrix_type not in self.unified_matrices:
                issues.append(f"Missing {matrix_type} matrices")
                continue
            
            matrices = self.unified_matrices[matrix_type]
            
            # Check required matrix components
            required_components = ['betas', 'weights', 'factor_covariance', 'descendant_leaves']
            for component in required_components:
                if component not in matrices:
                    issues.append(f"Missing {component} in {matrix_type} matrices")
        
        # Check matrix dimensions are consistent
        if len(issues) == 0:  # Only check if basic structure is valid
            try:
                port_betas = self.unified_matrices['portfolio']['betas']
                bench_betas = self.unified_matrices['benchmark']['betas']
                active_betas = self.unified_matrices['active']['betas']
                
                if port_betas.shape != bench_betas.shape:
                    issues.append(f"Portfolio and benchmark beta shapes don't match: {port_betas.shape} vs {bench_betas.shape}")
                
                if port_betas.shape != active_betas.shape:
                    issues.append(f"Portfolio and active beta shapes don't match: {port_betas.shape} vs {active_betas.shape}")
                
                # Check weights dimensions
                port_weights = self.unified_matrices['portfolio']['weights']
                bench_weights = self.unified_matrices['benchmark']['weights']
                
                if len(port_weights) != len(bench_weights):
                    issues.append(f"Portfolio and benchmark weight lengths don't match: {len(port_weights)} vs {len(bench_weights)}")
                
                if len(port_weights) != port_betas.shape[0]:
                    issues.append(f"Portfolio weights length {len(port_weights)} doesn't match beta rows {port_betas.shape[0]}")
                    
            except Exception as e:
                issues.append(f"Error validating matrix dimensions: {str(e)}")
        
        return {
            'passes': len(issues) == 0,
            'issues': issues,
            'message': '; '.join(issues) if issues else 'Validation passed'
        }
    
    def get_asset_names(self) -> List[str]:
        """Get descendant leaf asset names"""
        return list(self.descendant_leaves)
    
    def get_factor_names(self) -> List[str]:
        """Get factor names"""
        return list(self.factor_names)
    
    def _extract_factor_names(self) -> List[str]:
        """Extract factor names from factor_returns DataFrame in unified_matrices."""
        # Try to get factor names from portfolio factor_returns first
        for matrix_type in ['portfolio', 'benchmark', 'active']:
            if (matrix_type in self.unified_matrices and 
                'factor_returns' in self.unified_matrices[matrix_type]):
                factor_returns_df = self.unified_matrices[matrix_type]['factor_returns']
                if factor_returns_df is not None and hasattr(factor_returns_df, 'columns'):
                    return list(factor_returns_df.columns)
        
        # Fallback: if no factor_returns found, try to infer from factor_covariance dimensions
        if 'portfolio' in self.unified_matrices and 'factor_covariance' in self.unified_matrices['portfolio']:
            factor_covar = self.unified_matrices['portfolio']['factor_covariance']
            num_factors = factor_covar.shape[0]
            return [f'Factor_{i+1}' for i in range(num_factors)]
        
        # Final fallback: empty list (will likely cause validation error)
        return []
    
    def _create_risk_models(self) -> None:
        """Create LinearRiskModel instances for portfolio, benchmark, and active."""
        from .estimator import LinearRiskModel
        
        # Get residual variance matrices - use residual_covariance for consistency
        portfolio_resvar = self.unified_matrices['portfolio']['residual_covariance']
        benchmark_resvar = self.unified_matrices['benchmark']['residual_covariance']
        active_resvar = self.unified_matrices['active']['residual_covariance']
        
        # Ensure they are 2D matrices (convert from vector if needed)
        if portfolio_resvar.ndim == 1:
            portfolio_resvar = np.diag(portfolio_resvar)
        if benchmark_resvar.ndim == 1:
            benchmark_resvar = np.diag(benchmark_resvar)
        if active_resvar.ndim == 1:
            active_resvar = np.diag(active_resvar)
        
        self.portfolio_model = LinearRiskModel(
            beta=self.unified_matrices['portfolio']['betas'],
            factor_covar=self.unified_matrices['portfolio']['factor_covariance'],
            resvar=portfolio_resvar,
            frequency=self.unified_matrices['portfolio']['frequency'],
            symbols=self.descendant_leaves,
            factor_names=self.factor_names,
            asset_returns=self.unified_matrices['portfolio']['asset_returns'],
            factor_returns=self.unified_matrices['portfolio']['factor_returns'],
            residual_returns=self.unified_matrices['portfolio']['residual_returns']
        )
        
        self.benchmark_model = LinearRiskModel(
            beta=self.unified_matrices['benchmark']['betas'],
            factor_covar=self.unified_matrices['benchmark']['factor_covariance'],
            resvar=benchmark_resvar,
            frequency=self.unified_matrices['benchmark']['frequency'],
            symbols=self.descendant_leaves,
            factor_names=self.factor_names,
            asset_returns=self.unified_matrices['benchmark']['asset_returns'],
            factor_returns=self.unified_matrices['benchmark']['factor_returns'],
            residual_returns=self.unified_matrices['benchmark']['residual_returns']
        )
        
        self.active_model = LinearRiskModel(
            beta=self.unified_matrices['active']['betas'],
            factor_covar=self.unified_matrices['active']['factor_covariance'],
            resvar=active_resvar,
            frequency=self.unified_matrices['active']['frequency'],
            symbols=self.descendant_leaves,
            factor_names=self.factor_names,
            asset_returns=self.unified_matrices['active']['asset_returns'],
            factor_returns=self.unified_matrices['active']['factor_returns'],
            residual_returns=self.unified_matrices['active']['residual_returns']
        )
    
    def _create_decomposers(self) -> None:
        """Create decomposer instances for all risk types using new unified architecture."""
        from .decomposer import RiskDecomposer
        
        # Get weight vectors
        portfolio_weights = self.unified_matrices['portfolio']['weights']
        benchmark_weights = self.unified_matrices['benchmark']['weights']
        
        # Create individual decomposers for portfolio and benchmark using new architecture
        portfolio_context = create_single_model_context(
            self.portfolio_model, portfolio_weights, self.annualize
        )
        self.portfolio_decomposer = RiskDecomposer(portfolio_context)
        
        benchmark_context = create_single_model_context(
            self.benchmark_model, benchmark_weights, self.annualize
        )
        self.benchmark_decomposer = RiskDecomposer(benchmark_context)
        
        # Create comprehensive active risk decomposer using new architecture
        # Check if cross-covariance matrix is available in unified_matrices
        cross_covar = self.unified_matrices.get('cross_covariance', None)
        
        active_context = create_active_risk_context(
            self.portfolio_model, self.benchmark_model,
            portfolio_weights, benchmark_weights,
            self.active_model, cross_covar, self.annualize
        )
        self.active_decomposer = RiskDecomposer(active_context)
    
    def _store_results(self) -> None:
        """Store comprehensive results from all decomposers."""
        # Basic risk metrics for all types
        self.portfolio_total_risk = self.portfolio_decomposer.portfolio_volatility
        self.benchmark_total_risk = self.benchmark_decomposer.portfolio_volatility
        self.active_total_risk = self.active_decomposer.portfolio_volatility


def create_hierarchical_risk_context(
    unified_matrices: Dict[str, Dict[str, Any]], 
    annualize: bool = True
) -> HierarchicalModelContext:
    """
    Convenience function to create a hierarchical model context.
    
    Parameters
    ----------
    unified_matrices : Dict[str, Dict[str, Any]]
        Unified matrices containing 'portfolio', 'benchmark', and 'active' risk data.
        Factor names will be extracted from the factor_returns DataFrame.
    annualize : bool, default True
        Whether to annualize risk metrics
        
    Returns
    -------
    HierarchicalModelContext
        Configured hierarchical context
    """
    return HierarchicalModelContext(unified_matrices, annualize)


def create_single_model_context(
    model: RiskModel,
    weights: np.ndarray,
    annualize: bool = True
) -> SingleModelContext:
    """
    Convenience function to create a single-model context.
    
    Parameters
    ----------
    model : RiskModel
        Risk model
    weights : np.ndarray
        Portfolio weights
    annualize : bool, default True
        Whether to annualize metrics
        
    Returns
    -------
    SingleModelContext
        Configured context
    """
    return SingleModelContext(model, weights, annualize)


def create_active_risk_context(
    portfolio_model: RiskModel,
    benchmark_model: RiskModel,
    portfolio_weights: np.ndarray,
    benchmark_weights: np.ndarray,
    active_model: Optional[RiskModel] = None,
    cross_covar: Optional[np.ndarray] = None,
    annualize: bool = True
) -> MultiModelContext:
    """
    Convenience function to create an active risk context.
    
    Parameters
    ----------
    portfolio_model : RiskModel
        Portfolio risk model
    benchmark_model : RiskModel
        Benchmark risk model
    portfolio_weights : np.ndarray
        Portfolio weights
    benchmark_weights : np.ndarray
        Benchmark weights
    active_model : RiskModel, optional
        Active returns risk model
    cross_covar : np.ndarray, optional
        Cross-covariance matrix between benchmark and active returns
    annualize : bool, default True
        Whether to annualize metrics
        
    Returns
    -------
    MultiModelContext
        Configured context
    """
    return MultiModelContext(
        portfolio_model, benchmark_model,
        portfolio_weights, benchmark_weights,
        active_model, cross_covar, annualize
    )


# =========================================================================
# DECOMPOSER FACTORY FUNCTIONS
# =========================================================================

def create_portfolio_decomposer(model, weights, annualize: bool = True):
    """
    Create a portfolio risk decomposer.
    
    Parameters
    ----------
    model : RiskModel
        Risk model
    weights : array-like
        Portfolio weights
    annualize : bool, default True
        Whether to annualize metrics
        
    Returns
    -------
    RiskDecomposer
        Configured decomposer for portfolio analysis
    """
    from .decomposer import RiskDecomposer
    context = create_single_model_context(model, weights, annualize)
    return RiskDecomposer(context)


def create_active_risk_decomposer(
    portfolio_model, 
    benchmark_model,
    portfolio_weights,
    benchmark_weights,
    active_model=None,
    annualize: bool = True
):
    """
    Create an active risk decomposer.
    
    Parameters
    ----------
    portfolio_model : RiskModel
        Portfolio risk model
    benchmark_model : RiskModel
        Benchmark risk model
    portfolio_weights : array-like
        Portfolio weights
    benchmark_weights : array-like
        Benchmark weights
    active_model : RiskModel, optional
        Active returns risk model
    annualize : bool, default True
        Whether to annualize metrics
        
    Returns
    -------
    RiskDecomposer
        Configured decomposer for active risk analysis
    """
    from .decomposer import RiskDecomposer
    context = create_active_risk_context(
        portfolio_model, benchmark_model,
        portfolio_weights, benchmark_weights,
        active_model, annualize
    )
    return RiskDecomposer(context)