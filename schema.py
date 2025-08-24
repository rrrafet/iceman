"""
Unified Risk Result Schema
==========================

Standardized schema and validation for all risk analysis results across Spark.
This module provides a unified structure to eliminate overlaps and inconsistencies
between different risk analysis components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Type
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum


class AnalysisType(Enum):
    """Enumeration of supported analysis types"""
    PORTFOLIO = "portfolio"
    ACTIVE = "active" 
    HIERARCHICAL = "hierarchical"
    BENCHMARK = "benchmark"


class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"       # All validations must pass
    MODERATE = "moderate"   # Core validations must pass, warnings for others
    LENIENT = "lenient"     # Only critical validations required


class RiskResultSchema:
    """
    Unified risk result schema providing consistent structure across all risk analysis.
    
    This class defines the standard format for risk analysis results, including
    validation, conversion utilities, and compatibility methods.
    """
    
    def __init__(
        self,
        analysis_type: Union[AnalysisType, str],
        asset_names: Optional[List[str]] = None,
        factor_names: Optional[List[str]] = None,
        component_ids: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None,
        data_frequency: str = "D",
        annualized: bool = True,
        validation_level: ValidationLevel = ValidationLevel.MODERATE
    ):
        """
        Initialize unified risk result schema.
        
        Parameters
        ----------
        analysis_type : AnalysisType or str
            Type of risk analysis performed
        asset_names : list of str, optional
            Names/symbols of assets analyzed
        factor_names : list of str, optional
            Names of risk factors
        component_ids : list of str, optional
            Component identifiers for hierarchical analysis
        timestamp : datetime, optional
            Analysis timestamp, defaults to now
        data_frequency : str, default "D"
            Data frequency for annualization
        annualized : bool, default True
            Whether risk metrics are annualized
        validation_level : ValidationLevel, default MODERATE
            Validation strictness level
        """
        self.analysis_type = AnalysisType(analysis_type) if isinstance(analysis_type, str) else analysis_type
        self.asset_names = asset_names or []
        self.factor_names = factor_names or []
        self.component_ids = component_ids or []
        self.timestamp = timestamp or datetime.now()
        self.data_frequency = data_frequency
        self.annualized = annualized
        self.validation_level = validation_level
        
        # Initialize schema structure
        self._data = self._create_empty_schema()
    
    def _create_empty_schema(self) -> Dict[str, Any]:
        """Create empty schema structure with multi-lens architecture."""
        
        # Helper function to create lens-specific structure
        def _create_lens_structure():
            return {
                "core_metrics": {
                    "total_risk": None,
                    "factor_risk_contribution": None,
                    "specific_risk_contribution": None,
                    "factor_risk_percentage": None,
                    "specific_risk_percentage": None
                },
                "exposures": {
                    "factor_exposures": {},      # Named dict: {factor_name: exposure}
                    "factor_loadings": {}       # Asset-factor loadings: {asset_name: {factor_name: loading}}
                },
                "contributions": {
                    "by_asset": {},             # Named asset contributions: {asset_name: contribution}
                    "by_factor": {},            # Named factor contributions: {factor_name: contribution}
                    "by_component": {}          # For hierarchical: {component_id: contribution}
                },
                "matrices": {
                    "factor_risk_contributions": {},  # Asset × Factor matrix: {asset_name: {factor_name: contribution}}
                    "weighted_betas": {}              # Asset × Factor matrix: {asset_name: {factor_name: weighted_beta}}
                }
            }
        
        return {
            "metadata": {
                "analysis_type": self.analysis_type.value,
                "timestamp": self.timestamp.isoformat(),
                "data_frequency": self.data_frequency,
                "annualized": self.annualized,
                "schema_version": "2.0",  # Updated for multi-lens support
                "context_info": {}
            },
            "identifiers": {
                "asset_names": self.asset_names.copy(),
                "factor_names": self.factor_names.copy(),
                "component_ids": self.component_ids.copy()
            },
            
            # Hierarchical structure information
            "hierarchy": {
                "root_component": None,                    # Root component ID
                "component_relationships": {},             # {component_id: {"parent": parent_id, "children": [child_ids]}}
                "component_metadata": {},                  # {component_id: {"type": "node"/"leaf", "level": int, "path": str, ...}}
                "adjacency_list": {},                      # Graph adjacency representation {parent: [children]}
                "tree_structure": {},                      # Nested tree representation
                "traversal_order": [],                     # Component IDs in traversal order
                "leaf_components": [],                     # List of leaf component IDs
                "path_mappings": {},                       # {component_id: full_path_string}
                "level_mappings": {}                       # {level: [component_ids_at_level]}
            },
            
            # Time series data for all components
            "time_series": {
                "portfolio_returns": {},                       # {component_id: [return_series]}
                "benchmark_returns": {},                       # {component_id: [return_series]}
                "active_returns": {},                          # {component_id: [return_series]} 
                "factor_returns": {},                          # {factor_name: [return_series]}
                "dates": [],                                   # Date index for time series
                "frequency": None,                             # Data frequency (daily, monthly, etc.)
                "currency": None,                              # Currency for returns
                "metadata": {
                    "start_date": None,                        # First date in series
                    "end_date": None,                          # Last date in series
                    "total_periods": 0,                        # Number of periods
                    "missing_data_policy": "forward_fill",     # How missing data was handled
                    "return_type": "simple",                   # simple, log, excess
                    "annualization_factor": 252               # Business days for annualization
                },
                "statistics": {
                    "portfolio": {},                           # {component_id: {"mean": x, "std": y, "sharpe": z, ...}}
                    "benchmark": {},                           # Summary stats for benchmark returns
                    "active": {},                              # Summary stats for active returns
                    "factor": {}                               # Summary stats for factor returns
                },
                "correlations": {
                    "portfolio_vs_benchmark": {},             # {component_id: correlation_value}
                    "portfolio_vs_factors": {},               # {component_id: {factor_name: correlation}}
                    "factor_correlations": {},                # {factor1: {factor2: correlation}}
                    "hierarchical_correlations": {}           # Parent-child correlations within hierarchy
                }
            },
            
            # Multi-lens risk decomposition structure
            "portfolio": _create_lens_structure(),
            "benchmark": _create_lens_structure(),
            "active": {
                **_create_lens_structure(),
                "decomposition": {
                    "allocation_effect": {
                        "factor_contributions": {},      # Risk from factor exposure differences
                        "specific_contributions": {},    # Risk from specific exposure differences
                        "total_contribution": None
                    },
                    "selection_effect": {
                        "factor_contributions": {},      # Risk from asset selection within factors
                        "specific_contributions": {},    # Risk from specific asset selection
                        "total_contribution": None
                    },
                    "interaction_effect": {
                        "factor_contributions": {},      # Interaction between allocation and selection
                        "total_contribution": None
                    },
                    "validation": {
                        "total_decomposition_check": None,  # allocation + selection + interaction = total active
                        "component_sums_check": {}
                    }
                }
            },
            
            "weights": {
                "portfolio_weights": {},      # Named portfolio weights: {asset_name: weight}
                "benchmark_weights": {},      # Named benchmark weights: {asset_name: weight}
                "active_weights": {}          # Named active weights: {asset_name: weight}
            },
            
            # Backward compatibility sections (legacy flat structure)
            "core_metrics": {
                "total_risk": None,
                "factor_risk_contribution": None,
                "specific_risk_contribution": None,
                "factor_risk_percentage": None,
                "specific_risk_percentage": None
            },
            "exposures": {
                "factor_exposures": {},
                "factor_loadings": {}
            },
            "contributions": {
                "by_asset": {},
                "by_factor": {},
                "by_component": {}
            },
            "arrays": {
                "weights": {
                    "portfolio_weights": [],
                    "benchmark_weights": [],
                    "active_weights": []
                },
                "exposures": {},
                "contributions": {}
            },
            "matrices": {
                "factor_risk_contributions": {},
                "weighted_betas": {}
            },
            "active_risk": {},              # Legacy active risk metrics
            "validation": {
                "checks": {},
                "summary": "",
                "passes": True,
                "level": self.validation_level.value
            },
            "details": {}
        }
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get the complete schema data."""
        return self._data
    
    def set_core_metrics(
        self,
        total_risk: float,
        factor_risk_contribution: float,
        specific_risk_contribution: float
    ) -> None:
        """
        Set core risk metrics.
        
        Parameters
        ----------
        total_risk : float
            Total portfolio/active risk
        factor_risk_contribution : float
            Risk contribution from factors
        specific_risk_contribution : float
            Risk contribution from specific/idiosyncratic sources
        """
        self._data["core_metrics"]["total_risk"] = float(total_risk)
        self._data["core_metrics"]["factor_risk_contribution"] = float(factor_risk_contribution)
        self._data["core_metrics"]["specific_risk_contribution"] = float(specific_risk_contribution)
        
        # Calculate percentages
        if total_risk > 0:
            self._data["core_metrics"]["factor_risk_percentage"] = 100.0 * factor_risk_contribution / total_risk
            self._data["core_metrics"]["specific_risk_percentage"] = 100.0 * specific_risk_contribution / total_risk
        else:
            self._data["core_metrics"]["factor_risk_percentage"] = 0.0
            self._data["core_metrics"]["specific_risk_percentage"] = 0.0
    
    def set_factor_exposures(self, exposures: Union[np.ndarray, Dict[str, float], List[float]]) -> None:
        """
        Set factor exposures with automatic name mapping.
        
        Parameters
        ----------
        exposures : array-like or dict
            Factor exposures, either as array/list or pre-named dictionary
        """
        if isinstance(exposures, dict):
            self._data["exposures"]["factor_exposures"] = exposures.copy()
        else:
            exposures_array = np.asarray(exposures)
            if len(self.factor_names) == len(exposures_array):
                self._data["exposures"]["factor_exposures"] = {
                    name: float(value) for name, value in zip(self.factor_names, exposures_array)
                }
            else:
                # Fallback with generic names
                self._data["exposures"]["factor_exposures"] = {
                    f"factor_{i}": float(value) for i, value in enumerate(exposures_array)
                }
        
        # Store raw array for backward compatibility
        if "exposures" not in self._data["arrays"]:
            self._data["arrays"]["exposures"] = {}
        self._data["arrays"]["exposures"]["factor_exposures"] = list(exposures) if not isinstance(exposures, dict) else list(exposures.values())
    
    def set_factor_loadings(self, loadings: Union[np.ndarray, Dict[str, Dict[str, float]]]) -> None:
        """
        Set factor loadings (beta matrix) with automatic name mapping.
        
        Parameters
        ----------
        loadings : array-like or dict
            Factor loadings, either as N×K matrix or nested dictionary
        """
        if isinstance(loadings, dict):
            self._data["exposures"]["factor_loadings"] = loadings.copy()
        else:
            loadings_array = np.asarray(loadings)
            if loadings_array.ndim == 2:
                n_assets, n_factors = loadings_array.shape
                asset_names = self.asset_names if len(self.asset_names) == n_assets else [f"asset_{i}" for i in range(n_assets)]
                factor_names = self.factor_names if len(self.factor_names) == n_factors else [f"factor_{i}" for i in range(n_factors)]
                
                self._data["exposures"]["factor_loadings"] = {
                    asset_name: {
                        factor_name: float(loadings_array[i, j])
                        for j, factor_name in enumerate(factor_names)
                    }
                    for i, asset_name in enumerate(asset_names)
                }
            else:
                raise ValueError("Factor loadings must be 2D array (N assets × K factors) or nested dictionary")
        
        # Store raw array for backward compatibility
        if isinstance(loadings, np.ndarray):
            self._data["arrays"]["exposures"]["factor_loadings"] = loadings.tolist()
    
    def set_asset_contributions(self, contributions: Union[np.ndarray, Dict[str, float], List[float]]) -> None:
        """
        Set asset risk contributions with automatic name mapping.
        
        Parameters
        ----------
        contributions : array-like or dict
            Asset contributions, either as array/list or pre-named dictionary
        """
        if isinstance(contributions, dict):
            self._data["contributions"]["by_asset"] = contributions.copy()
        else:
            contributions_array = np.asarray(contributions)
            if len(self.asset_names) == len(contributions_array):
                self._data["contributions"]["by_asset"] = {
                    name: float(value) for name, value in zip(self.asset_names, contributions_array)
                }
            else:
                # Fallback with generic names
                self._data["contributions"]["by_asset"] = {
                    f"asset_{i}": float(value) for i, value in enumerate(contributions_array)
                }
        
        # Store raw array for backward compatibility
        if "contributions" not in self._data["arrays"]:
            self._data["arrays"]["contributions"] = {}
        self._data["arrays"]["contributions"]["asset_contributions"] = list(contributions) if not isinstance(contributions, dict) else list(contributions.values())
    
    def set_factor_contributions(self, contributions: Union[np.ndarray, Dict[str, float], List[float]]) -> None:
        """
        Set factor risk contributions with automatic name mapping.
        
        Parameters
        ----------
        contributions : array-like or dict
            Factor contributions, either as array/list or pre-named dictionary
        """
        if isinstance(contributions, dict):
            self._data["contributions"]["by_factor"] = contributions.copy()
        else:
            contributions_array = np.asarray(contributions)
            if len(self.factor_names) == len(contributions_array):
                self._data["contributions"]["by_factor"] = {
                    name: float(value) for name, value in zip(self.factor_names, contributions_array)
                }
            else:
                # Fallback with generic names
                self._data["contributions"]["by_factor"] = {
                    f"factor_{i}": float(value) for i, value in enumerate(contributions_array)
                }
        
        # Store raw array for backward compatibility
        self._data["arrays"]["contributions"]["factor_contributions"] = list(contributions) if not isinstance(contributions, dict) else list(contributions.values())
    
    def set_portfolio_weights(self, weights: Union[np.ndarray, Dict[str, float], List[float]]) -> None:
        """
        Set portfolio weights with automatic name mapping.
        
        Parameters
        ----------
        weights : array-like or dict
            Portfolio weights, either as array/list or pre-named dictionary
        """
        if isinstance(weights, dict):
            self._data["weights"]["portfolio_weights"] = weights.copy()
        else:
            weights_array = np.asarray(weights)
            if len(self.asset_names) == len(weights_array):
                self._data["weights"]["portfolio_weights"] = {
                    name: float(value) for name, value in zip(self.asset_names, weights_array)
                }
            else:
                # Fallback with generic names
                self._data["weights"]["portfolio_weights"] = {
                    f"asset_{i}": float(value) for i, value in enumerate(weights_array)
                }
        
        # Store raw array for backward compatibility
        if "weights" not in self._data["arrays"]:
            self._data["arrays"]["weights"] = {}
        self._data["arrays"]["weights"]["portfolio_weights"] = list(weights) if not isinstance(weights, dict) else list(weights.values())
    
    def set_benchmark_weights(self, weights: Union[np.ndarray, Dict[str, float], List[float]]) -> None:
        """
        Set benchmark weights with automatic name mapping.
        
        Parameters
        ----------
        weights : array-like or dict
            Benchmark weights, either as array/list or pre-named dictionary
        """
        if isinstance(weights, dict):
            self._data["weights"]["benchmark_weights"] = weights.copy()
        else:
            weights_array = np.asarray(weights)
            if len(self.asset_names) == len(weights_array):
                self._data["weights"]["benchmark_weights"] = {
                    name: float(value) for name, value in zip(self.asset_names, weights_array)
                }
            else:
                # Fallback with generic names
                self._data["weights"]["benchmark_weights"] = {
                    f"asset_{i}": float(value) for i, value in enumerate(weights_array)
                }
        
        # Store raw array for backward compatibility
        self._data["arrays"]["weights"]["benchmark_weights"] = list(weights) if not isinstance(weights, dict) else list(weights.values())
    
    def set_active_weights(self, weights: Optional[Union[np.ndarray, Dict[str, float], List[float]]] = None, 
                          auto_calculate: bool = True) -> None:
        """
        Set active weights with automatic name mapping.
        
        Parameters
        ----------
        weights : array-like or dict, optional
            Active weights, either as array/list or pre-named dictionary.
            If None and auto_calculate=True, will calculate from portfolio - benchmark
        auto_calculate : bool, default True
            If True and weights is None, automatically calculate active weights
            from portfolio_weights - benchmark_weights
        """
        if weights is None and auto_calculate:
            # Try to calculate active weights from existing portfolio and benchmark weights
            portfolio_weights = self._data["weights"]["portfolio_weights"]
            benchmark_weights = self._data["weights"]["benchmark_weights"]
            
            if portfolio_weights and benchmark_weights:
                # Calculate active weights from named dictionaries
                all_assets = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
                active_weights_dict = {}
                for asset in all_assets:
                    port_weight = portfolio_weights.get(asset, 0.0)
                    bench_weight = benchmark_weights.get(asset, 0.0)
                    active_weights_dict[asset] = port_weight - bench_weight
                
                self._data["weights"]["active_weights"] = active_weights_dict
                self._data["arrays"]["weights"]["active_weights"] = list(active_weights_dict.values())
                return
            else:
                raise ValueError("Cannot auto-calculate active weights: portfolio or benchmark weights not set")
        
        if weights is None:
            raise ValueError("weights parameter cannot be None when auto_calculate=False")
        
        # Set weights explicitly
        if isinstance(weights, dict):
            self._data["weights"]["active_weights"] = weights.copy()
        else:
            weights_array = np.asarray(weights)
            if len(self.asset_names) == len(weights_array):
                self._data["weights"]["active_weights"] = {
                    name: float(value) for name, value in zip(self.asset_names, weights_array)
                }
            else:
                # Fallback with generic names
                self._data["weights"]["active_weights"] = {
                    f"asset_{i}": float(value) for i, value in enumerate(weights_array)
                }
        
        # Store raw array for backward compatibility
        self._data["arrays"]["weights"]["active_weights"] = list(weights) if not isinstance(weights, dict) else list(weights.values())
    
    # Multi-lens setter methods
    def set_lens_core_metrics(
        self, 
        lens: str,
        total_risk: float,
        factor_risk_contribution: float,
        specific_risk_contribution: float
    ) -> None:
        """
        Set core risk metrics for a specific lens (portfolio, benchmark, or active).
        
        Parameters
        ----------
        lens : str
            Lens type: 'portfolio', 'benchmark', or 'active'
        total_risk : float
            Total risk for this lens
        factor_risk_contribution : float
            Risk contribution from factors
        specific_risk_contribution : float
            Risk contribution from specific/idiosyncratic sources
        """
        if lens not in ['portfolio', 'benchmark', 'active']:
            raise ValueError("lens must be 'portfolio', 'benchmark', or 'active'")
        
        self._data[lens]["core_metrics"]["total_risk"] = float(total_risk)
        self._data[lens]["core_metrics"]["factor_risk_contribution"] = float(factor_risk_contribution)
        self._data[lens]["core_metrics"]["specific_risk_contribution"] = float(specific_risk_contribution)
        
        # Calculate percentages
        if total_risk > 0:
            self._data[lens]["core_metrics"]["factor_risk_percentage"] = 100.0 * factor_risk_contribution / total_risk
            self._data[lens]["core_metrics"]["specific_risk_percentage"] = 100.0 * specific_risk_contribution / total_risk
        else:
            self._data[lens]["core_metrics"]["factor_risk_percentage"] = 0.0
            self._data[lens]["core_metrics"]["specific_risk_percentage"] = 0.0
    
    def set_lens_factor_exposures(self, lens: str, exposures: Union[np.ndarray, Dict[str, float], List[float]]) -> None:
        """
        Set factor exposures for a specific lens.
        
        Parameters
        ----------
        lens : str
            Lens type: 'portfolio', 'benchmark', or 'active'
        exposures : array-like or dict
            Factor exposures, either as array/list or pre-named dictionary
        """
        if lens not in ['portfolio', 'benchmark', 'active']:
            raise ValueError("lens must be 'portfolio', 'benchmark', or 'active'")
            
        if isinstance(exposures, dict):
            self._data[lens]["exposures"]["factor_exposures"] = exposures.copy()
        else:
            exposures_array = np.asarray(exposures)
            if len(self.factor_names) == len(exposures_array):
                self._data[lens]["exposures"]["factor_exposures"] = {
                    name: float(value) for name, value in zip(self.factor_names, exposures_array)
                }
            else:
                # Fallback with generic names
                self._data[lens]["exposures"]["factor_exposures"] = {
                    f"factor_{i}": float(value) for i, value in enumerate(exposures_array)
                }
    
    def set_lens_asset_contributions(self, lens: str, contributions: Union[np.ndarray, Dict[str, float], List[float]]) -> None:
        """
        Set asset risk contributions for a specific lens.
        
        Parameters
        ----------
        lens : str
            Lens type: 'portfolio', 'benchmark', or 'active'
        contributions : array-like or dict
            Asset contributions, either as array/list or pre-named dictionary
        """
        if lens not in ['portfolio', 'benchmark', 'active']:
            raise ValueError("lens must be 'portfolio', 'benchmark', or 'active'")
            
        if isinstance(contributions, dict):
            self._data[lens]["contributions"]["by_asset"] = contributions.copy()
        else:
            contributions_array = np.asarray(contributions)
            if len(self.asset_names) == len(contributions_array):
                self._data[lens]["contributions"]["by_asset"] = {
                    name: float(value) for name, value in zip(self.asset_names, contributions_array)
                }
            else:
                # Fallback with generic names
                self._data[lens]["contributions"]["by_asset"] = {
                    f"asset_{i}": float(value) for i, value in enumerate(contributions_array)
                }
    
    def set_lens_factor_contributions(self, lens: str, contributions: Union[np.ndarray, Dict[str, float], List[float]]) -> None:
        """
        Set factor risk contributions for a specific lens.
        
        Parameters
        ----------
        lens : str
            Lens type: 'portfolio', 'benchmark', or 'active'
        contributions : array-like or dict
            Factor contributions, either as array/list or pre-named dictionary
        """
        if lens not in ['portfolio', 'benchmark', 'active']:
            raise ValueError("lens must be 'portfolio', 'benchmark', or 'active'")
            
        if isinstance(contributions, dict):
            self._data[lens]["contributions"]["by_factor"] = contributions.copy()
        else:
            contributions_array = np.asarray(contributions)
            if len(self.factor_names) == len(contributions_array):
                self._data[lens]["contributions"]["by_factor"] = {
                    name: float(value) for name, value in zip(self.factor_names, contributions_array)
                }
            else:
                # Fallback with generic names
                self._data[lens]["contributions"]["by_factor"] = {
                    f"factor_{i}": float(value) for i, value in enumerate(contributions_array)
                }
    
    def set_lens_factor_risk_contributions_matrix(self, lens: str, matrix: Union[np.ndarray, Dict[str, Dict[str, float]]]) -> None:
        """
        Set factor risk contributions matrix for a specific lens.
        
        Parameters
        ----------
        lens : str
            Lens type: 'portfolio', 'benchmark', or 'active'
        matrix : array-like or dict
            Factor risk contributions matrix, either as N×K array or nested dictionary
        """
        if lens not in ['portfolio', 'benchmark', 'active']:
            raise ValueError("lens must be 'portfolio', 'benchmark', or 'active'")
            
        if isinstance(matrix, dict):
            self._data[lens]["matrices"]["factor_risk_contributions"] = matrix.copy()
        else:
            matrix_array = np.asarray(matrix)
            if matrix_array.ndim == 2:
                n_assets, n_factors = matrix_array.shape
                asset_names = self.asset_names if len(self.asset_names) == n_assets else [f"asset_{i}" for i in range(n_assets)]
                factor_names = self.factor_names if len(self.factor_names) == n_factors else [f"factor_{i}" for i in range(n_factors)]
                
                self._data[lens]["matrices"]["factor_risk_contributions"] = {
                    asset_name: {
                        factor_name: float(matrix_array[i, j])
                        for j, factor_name in enumerate(factor_names)
                    }
                    for i, asset_name in enumerate(asset_names)
                }
            else:
                raise ValueError("Factor risk contributions matrix must be 2D array (N assets × K factors) or nested dictionary")
    
    def set_active_decomposition(
        self,
        allocation_factor_contrib: Optional[Dict[str, float]] = None,
        allocation_specific_contrib: Optional[Dict[str, float]] = None,
        selection_factor_contrib: Optional[Dict[str, float]] = None,
        selection_specific_contrib: Optional[Dict[str, float]] = None,
        interaction_factor_contrib: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Set active risk decomposition into allocation, selection, and interaction effects.
        
        Parameters
        ----------
        allocation_factor_contrib : dict, optional
            Factor contributions from allocation effect (different factor exposures)
        allocation_specific_contrib : dict, optional
            Specific contributions from allocation effect
        selection_factor_contrib : dict, optional
            Factor contributions from selection effect (asset selection within factors)
        selection_specific_contrib : dict, optional
            Specific contributions from selection effect
        interaction_factor_contrib : dict, optional
            Factor contributions from interaction between allocation and selection
        """
        decomp = self._data["active"]["decomposition"]
        
        # Allocation effect
        if allocation_factor_contrib is not None:
            decomp["allocation_effect"]["factor_contributions"] = allocation_factor_contrib.copy()
            decomp["allocation_effect"]["total_contribution"] = sum(allocation_factor_contrib.values())
        
        if allocation_specific_contrib is not None:
            decomp["allocation_effect"]["specific_contributions"] = allocation_specific_contrib.copy()
            if decomp["allocation_effect"]["total_contribution"] is not None:
                decomp["allocation_effect"]["total_contribution"] += sum(allocation_specific_contrib.values())
            else:
                decomp["allocation_effect"]["total_contribution"] = sum(allocation_specific_contrib.values())
        
        # Selection effect
        if selection_factor_contrib is not None:
            decomp["selection_effect"]["factor_contributions"] = selection_factor_contrib.copy()
            decomp["selection_effect"]["total_contribution"] = sum(selection_factor_contrib.values())
        
        if selection_specific_contrib is not None:
            decomp["selection_effect"]["specific_contributions"] = selection_specific_contrib.copy()
            if decomp["selection_effect"]["total_contribution"] is not None:
                decomp["selection_effect"]["total_contribution"] += sum(selection_specific_contrib.values())
            else:
                decomp["selection_effect"]["total_contribution"] = sum(selection_specific_contrib.values())
        
        # Interaction effect
        if interaction_factor_contrib is not None:
            decomp["interaction_effect"]["factor_contributions"] = interaction_factor_contrib.copy()
            decomp["interaction_effect"]["total_contribution"] = sum(interaction_factor_contrib.values())
        
        # Validate decomposition if all components are available
        self._validate_active_decomposition()
    
    def _validate_active_decomposition(self) -> None:
        """Validate that active decomposition components sum to total active risk."""
        decomp = self._data["active"]["decomposition"]
        active_core = self._data["active"]["core_metrics"]
        
        # Check if we have total active risk to validate against
        total_active_risk = active_core.get("total_risk")
        if total_active_risk is None:
            return
        
        # Sum up all decomposition components
        allocation_total = decomp["allocation_effect"].get("total_contribution", 0.0)
        selection_total = decomp["selection_effect"].get("total_contribution", 0.0)
        interaction_total = decomp["interaction_effect"].get("total_contribution", 0.0)
        
        decomposition_sum = allocation_total + selection_total + interaction_total
        difference = abs(decomposition_sum - total_active_risk)
        
        decomp["validation"]["total_decomposition_check"] = {
            "passes": difference < 1e-6,
            "expected": total_active_risk,
            "actual": decomposition_sum,
            "difference": difference,
            "components": {
                "allocation": allocation_total,
                "selection": selection_total,
                "interaction": interaction_total
            }
        }
    
    def set_hierarchy_structure(
        self,
        root_component: str,
        component_relationships: Dict[str, Dict[str, Any]],
        component_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        adjacency_list: Optional[Dict[str, List[str]]] = None,
        tree_structure: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set the hierarchical structure of the portfolio.
        
        Parameters
        ----------
        root_component : str
            Root component identifier
        component_relationships : dict
            Component parent/child relationships: {component_id: {"parent": parent_id, "children": [child_ids]}}
        component_metadata : dict, optional
            Component metadata: {component_id: {"type": "node"/"leaf", "level": int, "path": str, ...}}
        adjacency_list : dict, optional
            Graph adjacency representation: {parent: [children]}
        tree_structure : dict, optional
            Nested tree representation
        """
        hierarchy = self._data["hierarchy"]
        
        hierarchy["root_component"] = root_component
        hierarchy["component_relationships"] = component_relationships.copy()
        
        if component_metadata:
            hierarchy["component_metadata"] = component_metadata.copy()
        
        if adjacency_list:
            hierarchy["adjacency_list"] = adjacency_list.copy()
        
        if tree_structure:
            hierarchy["tree_structure"] = tree_structure.copy()
        
        # Auto-generate derived information
        self._generate_hierarchy_derived_info()
    
    def _generate_hierarchy_derived_info(self) -> None:
        """Generate derived hierarchy information like paths, levels, and leaf components."""
        hierarchy = self._data["hierarchy"]
        relationships = hierarchy["component_relationships"]
        metadata = hierarchy["component_metadata"]
        
        # Generate leaf components list
        leaf_components = []
        for component_id, relationship in relationships.items():
            if not relationship.get("children") or len(relationship["children"]) == 0:
                leaf_components.append(component_id)
        hierarchy["leaf_components"] = leaf_components
        
        # Generate path mappings from metadata
        path_mappings = {}
        level_mappings = {}
        for component_id, meta in metadata.items():
            if "path" in meta:
                path_mappings[component_id] = meta["path"]
            if "level" in meta:
                level = meta["level"]
                if level not in level_mappings:
                    level_mappings[level] = []
                level_mappings[level].append(component_id)
        
        hierarchy["path_mappings"] = path_mappings
        hierarchy["level_mappings"] = level_mappings
        
        # Generate traversal order (breadth-first)
        if hierarchy["root_component"]:
            traversal_order = self._generate_traversal_order(hierarchy["root_component"], relationships)
            hierarchy["traversal_order"] = traversal_order
    
    def _generate_traversal_order(self, root: str, relationships: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate breadth-first traversal order of the hierarchy."""
        traversal = []
        queue = [root]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                traversal.append(current)
                
                # Add children to queue
                if current in relationships and "children" in relationships[current]:
                    children = relationships[current]["children"]
                    if children:
                        queue.extend(children)
        
        return traversal
    
    def add_component_relationship(self, component_id: str, parent_id: Optional[str] = None, 
                                 children: Optional[List[str]] = None) -> None:
        """
        Add or update a single component's relationships.
        
        Parameters
        ----------
        component_id : str
            Component identifier
        parent_id : str, optional
            Parent component identifier
        children : list of str, optional
            List of child component identifiers
        """
        hierarchy = self._data["hierarchy"]
        
        if "component_relationships" not in hierarchy:
            hierarchy["component_relationships"] = {}
        
        relationship = hierarchy["component_relationships"].get(component_id, {})
        
        if parent_id is not None:
            relationship["parent"] = parent_id
        
        if children is not None:
            relationship["children"] = children.copy()
        
        hierarchy["component_relationships"][component_id] = relationship
        
        # Regenerate derived info
        self._generate_hierarchy_derived_info()
    
    def add_component_metadata(self, component_id: str, metadata: Dict[str, Any]) -> None:
        """
        Add or update metadata for a component.
        
        Parameters
        ----------
        component_id : str
            Component identifier
        metadata : dict
            Metadata dictionary (e.g., {"type": "leaf", "level": 3, "path": "portfolio/equity/us"})
        """
        hierarchy = self._data["hierarchy"]
        
        if "component_metadata" not in hierarchy:
            hierarchy["component_metadata"] = {}
        
        hierarchy["component_metadata"][component_id] = metadata.copy()
        
        # Regenerate derived info
        self._generate_hierarchy_derived_info()
    
    def get_hierarchy_summary(self) -> Dict[str, Any]:
        """Get a summary of the hierarchical structure."""
        hierarchy = self._data["hierarchy"]
        
        return {
            "root_component": hierarchy.get("root_component"),
            "total_components": len(hierarchy.get("component_relationships", {})),
            "leaf_components": len(hierarchy.get("leaf_components", [])),
            "max_depth": max(hierarchy.get("level_mappings", {}).keys()) if hierarchy.get("level_mappings") else 0,
            "components_by_level": {level: len(components) for level, components in hierarchy.get("level_mappings", {}).items()},
            "has_complete_structure": bool(
                hierarchy.get("root_component") and 
                hierarchy.get("component_relationships") and 
                hierarchy.get("component_metadata")
            )
        }
    
    # Time Series Methods
    def set_time_series_metadata(
        self,
        dates: List[Any],
        frequency: str = "daily",
        currency: str = "USD",
        return_type: str = "simple",
        annualization_factor: int = 252
    ) -> None:
        """
        Set time series metadata and date index.
        
        Parameters
        ----------
        dates : list
            Date index for the time series
        frequency : str, default "daily"
            Data frequency (daily, monthly, quarterly, annual)
        currency : str, default "USD"
            Currency for returns
        return_type : str, default "simple"
            Type of returns (simple, log, excess)
        annualization_factor : int, default 252
            Number of periods for annualization
        """
        self._data["time_series"]["dates"] = list(dates)
        self._data["time_series"]["frequency"] = frequency
        self._data["time_series"]["currency"] = currency
        
        metadata = self._data["time_series"]["metadata"]
        if len(dates) > 0:
            metadata["start_date"] = dates[0] if hasattr(dates[0], 'strftime') else str(dates[0])
            metadata["end_date"] = dates[-1] if hasattr(dates[-1], 'strftime') else str(dates[-1])
            metadata["total_periods"] = len(dates)
        
        metadata["return_type"] = return_type
        metadata["annualization_factor"] = annualization_factor
    
    def set_component_portfolio_returns(
        self,
        component_id: str,
        returns: Union[List[float], np.ndarray, pd.Series]
    ) -> None:
        """
        Set portfolio return time series for a specific component.
        
        Parameters
        ----------
        component_id : str
            ID of the component (node or leaf)
        returns : array-like
            Time series of portfolio returns for this component
        """
        if isinstance(returns, pd.Series):
            return_values = returns.values.tolist()
        else:
            return_values = np.asarray(returns).tolist()
        
        self._data["time_series"]["portfolio_returns"][component_id] = return_values
        
        # Calculate and store basic statistics
        self._calculate_component_statistics(component_id, return_values, "portfolio")
    
    def set_component_benchmark_returns(
        self,
        component_id: str,
        returns: Union[List[float], np.ndarray, pd.Series]
    ) -> None:
        """
        Set benchmark return time series for a specific component.
        
        Parameters
        ----------
        component_id : str
            ID of the component (node or leaf)
        returns : array-like
            Time series of benchmark returns for this component
        """
        if isinstance(returns, pd.Series):
            return_values = returns.values.tolist()
        else:
            return_values = np.asarray(returns).tolist()
        
        self._data["time_series"]["benchmark_returns"][component_id] = return_values
        
        # Calculate and store basic statistics
        self._calculate_component_statistics(component_id, return_values, "benchmark")
        
        # Calculate active returns if portfolio returns exist
        if component_id in self._data["time_series"]["portfolio_returns"]:
            self._calculate_active_returns(component_id)
    
    def set_component_active_returns(
        self,
        component_id: str,
        returns: Union[List[float], np.ndarray, pd.Series]
    ) -> None:
        """
        Set active return time series for a specific component.
        
        Parameters
        ----------
        component_id : str
            ID of the component (node or leaf)
        returns : array-like
            Time series of active returns for this component
        """
        if isinstance(returns, pd.Series):
            return_values = returns.values.tolist()
        else:
            return_values = np.asarray(returns).tolist()
        
        self._data["time_series"]["active_returns"][component_id] = return_values
        
        # Calculate and store basic statistics
        self._calculate_component_statistics(component_id, return_values, "active")
    
    def set_factor_returns(
        self,
        factor_name: str,
        returns: Union[List[float], np.ndarray, pd.Series]
    ) -> None:
        """
        Set return time series for a risk factor.
        
        Parameters
        ----------
        factor_name : str
            Name of the risk factor
        returns : array-like
            Time series of factor returns
        """
        if isinstance(returns, pd.Series):
            return_values = returns.values.tolist()
        else:
            return_values = np.asarray(returns).tolist()
        
        self._data["time_series"]["factor_returns"][factor_name] = return_values
        
        # Calculate and store basic statistics
        self._calculate_factor_statistics(factor_name, return_values)
    
    def set_multiple_component_returns(
        self,
        returns_dict: Dict[str, Dict[str, Union[List[float], np.ndarray, pd.Series]]]
    ) -> None:
        """
        Set time series for multiple components at once.
        
        Parameters
        ----------
        returns_dict : dict
            Nested dictionary with structure:
            {
                "portfolio": {component_id: returns, ...},
                "benchmark": {component_id: returns, ...},
                "active": {component_id: returns, ...},
                "factor": {factor_name: returns, ...}
            }
        """
        for return_type, components in returns_dict.items():
            for component_id, returns in components.items():
                if return_type == "portfolio":
                    self.set_component_portfolio_returns(component_id, returns)
                elif return_type == "benchmark":
                    self.set_component_benchmark_returns(component_id, returns)
                elif return_type == "active":
                    self.set_component_active_returns(component_id, returns)
                elif return_type == "factor":
                    self.set_factor_returns(component_id, returns)
    
    def _calculate_component_statistics(
        self,
        component_id: str,
        returns: List[float],
        return_type: str
    ) -> None:
        """Calculate basic statistics for component returns."""
        if not returns:
            return
        
        returns_array = np.array(returns)
        valid_returns = returns_array[~np.isnan(returns_array)]
        
        if len(valid_returns) == 0:
            return
        
        stats = {
            "mean": float(np.mean(valid_returns)),
            "std": float(np.std(valid_returns, ddof=1)) if len(valid_returns) > 1 else 0.0,
            "min": float(np.min(valid_returns)),
            "max": float(np.max(valid_returns)),
            "skew": float(self._calculate_skewness(valid_returns)),
            "kurtosis": float(self._calculate_kurtosis(valid_returns)),
            "total_periods": len(returns),
            "valid_periods": len(valid_returns)
        }
        
        # Annualized metrics
        annualization_factor = self._data["time_series"]["metadata"]["annualization_factor"]
        if stats["std"] > 0:
            stats["annualized_return"] = stats["mean"] * annualization_factor
            stats["annualized_volatility"] = stats["std"] * np.sqrt(annualization_factor)
            stats["sharpe_ratio"] = stats["annualized_return"] / stats["annualized_volatility"]
        else:
            stats["annualized_return"] = 0.0
            stats["annualized_volatility"] = 0.0
            stats["sharpe_ratio"] = 0.0
        
        self._data["time_series"]["statistics"][return_type][component_id] = stats
    
    def _calculate_factor_statistics(
        self,
        factor_name: str,
        returns: List[float]
    ) -> None:
        """Calculate basic statistics for factor returns."""
        if not returns:
            return
        
        returns_array = np.array(returns)
        valid_returns = returns_array[~np.isnan(returns_array)]
        
        if len(valid_returns) == 0:
            return
        
        stats = {
            "mean": float(np.mean(valid_returns)),
            "std": float(np.std(valid_returns, ddof=1)) if len(valid_returns) > 1 else 0.0,
            "min": float(np.min(valid_returns)),
            "max": float(np.max(valid_returns)),
            "skew": float(self._calculate_skewness(valid_returns)),
            "kurtosis": float(self._calculate_kurtosis(valid_returns)),
            "total_periods": len(returns),
            "valid_periods": len(valid_returns)
        }
        
        # Annualized metrics
        annualization_factor = self._data["time_series"]["metadata"]["annualization_factor"]
        if stats["std"] > 0:
            stats["annualized_return"] = stats["mean"] * annualization_factor
            stats["annualized_volatility"] = stats["std"] * np.sqrt(annualization_factor)
        else:
            stats["annualized_return"] = 0.0
            stats["annualized_volatility"] = 0.0
        
        self._data["time_series"]["statistics"]["factor"][factor_name] = stats
    
    def _calculate_active_returns(self, component_id: str) -> None:
        """Calculate active returns as portfolio - benchmark for a component."""
        portfolio_returns = self._data["time_series"]["portfolio_returns"].get(component_id, [])
        benchmark_returns = self._data["time_series"]["benchmark_returns"].get(component_id, [])
        
        if portfolio_returns and benchmark_returns and len(portfolio_returns) == len(benchmark_returns):
            active_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
            self._data["time_series"]["active_returns"][component_id] = active_returns
            self._calculate_component_statistics(component_id, active_returns, "active")
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std == 0:
            return 0.0
        
        n = len(returns)
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((returns - mean) / std) ** 3)
        return skew
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis of returns."""
        if len(returns) < 4:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std == 0:
            return 0.0
        
        n = len(returns)
        kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((returns - mean) / std) ** 4)
        kurt = kurt - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))  # Excess kurtosis
        return kurt
    
    def calculate_correlations(self) -> None:
        """Calculate various correlation metrics between time series."""
        # Portfolio vs benchmark correlations
        for component_id in self._data["time_series"]["portfolio_returns"]:
            if component_id in self._data["time_series"]["benchmark_returns"]:
                port_returns = np.array(self._data["time_series"]["portfolio_returns"][component_id])
                bench_returns = np.array(self._data["time_series"]["benchmark_returns"][component_id])
                
                if len(port_returns) > 1 and len(bench_returns) > 1:
                    correlation = np.corrcoef(port_returns, bench_returns)[0, 1]
                    if not np.isnan(correlation):
                        self._data["time_series"]["correlations"]["portfolio_vs_benchmark"][component_id] = float(correlation)
        
        # Portfolio vs factor correlations
        for component_id in self._data["time_series"]["portfolio_returns"]:
            port_returns = np.array(self._data["time_series"]["portfolio_returns"][component_id])
            
            if component_id not in self._data["time_series"]["correlations"]["portfolio_vs_factors"]:
                self._data["time_series"]["correlations"]["portfolio_vs_factors"][component_id] = {}
            
            for factor_name, factor_returns in self._data["time_series"]["factor_returns"].items():
                factor_array = np.array(factor_returns)
                
                if len(port_returns) > 1 and len(factor_array) > 1 and len(port_returns) == len(factor_array):
                    correlation = np.corrcoef(port_returns, factor_array)[0, 1]
                    if not np.isnan(correlation):
                        self._data["time_series"]["correlations"]["portfolio_vs_factors"][component_id][factor_name] = float(correlation)
        
        # Factor-factor correlations
        factor_names = list(self._data["time_series"]["factor_returns"].keys())
        for i, factor1 in enumerate(factor_names):
            if factor1 not in self._data["time_series"]["correlations"]["factor_correlations"]:
                self._data["time_series"]["correlations"]["factor_correlations"][factor1] = {}
            
            for j, factor2 in enumerate(factor_names[i+1:], i+1):
                returns1 = np.array(self._data["time_series"]["factor_returns"][factor1])
                returns2 = np.array(self._data["time_series"]["factor_returns"][factor2])
                
                if len(returns1) > 1 and len(returns2) > 1 and len(returns1) == len(returns2):
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    if not np.isnan(correlation):
                        self._data["time_series"]["correlations"]["factor_correlations"][factor1][factor2] = float(correlation)
    
    def get_component_returns(
        self,
        component_id: str,
        return_type: str = "portfolio"
    ) -> List[float]:
        """
        Get return time series for a specific component.
        
        Parameters
        ----------
        component_id : str
            ID of the component
        return_type : str, default "portfolio"
            Type of returns ("portfolio", "benchmark", "active")
            
        Returns
        -------
        list of float
            Time series of returns
        """
        if return_type == "portfolio":
            return self._data["time_series"]["portfolio_returns"].get(component_id, [])
        elif return_type == "benchmark":
            return self._data["time_series"]["benchmark_returns"].get(component_id, [])
        elif return_type == "active":
            return self._data["time_series"]["active_returns"].get(component_id, [])
        else:
            return []
    
    def get_factor_returns(self, factor_name: str) -> List[float]:
        """
        Get return time series for a specific factor.
        
        Parameters
        ----------
        factor_name : str
            Name of the factor
            
        Returns
        -------
        list of float
            Time series of factor returns
        """
        return self._data["time_series"]["factor_returns"].get(factor_name, [])
    
    def get_time_series_summary(self) -> Dict[str, Any]:
        """Get a summary of all time series data."""
        ts_data = self._data["time_series"]
        
        return {
            "metadata": ts_data["metadata"].copy(),
            "component_counts": {
                "portfolio_components": len(ts_data["portfolio_returns"]),
                "benchmark_components": len(ts_data["benchmark_returns"]),
                "active_components": len(ts_data["active_returns"]),
                "factors": len(ts_data["factor_returns"])
            },
            "date_info": {
                "total_dates": len(ts_data["dates"]),
                "frequency": ts_data["frequency"],
                "currency": ts_data["currency"]
            },
            "statistics_available": {
                "portfolio": len(ts_data["statistics"]["portfolio"]),
                "benchmark": len(ts_data["statistics"]["benchmark"]),
                "active": len(ts_data["statistics"]["active"]),
                "factor": len(ts_data["statistics"]["factor"])
            }
        }
    
    def set_factor_risk_contributions_matrix(self, matrix: Union[np.ndarray, Dict[str, Dict[str, float]]]) -> None:
        """
        Set factor risk contributions matrix (Asset × Factor).
        
        Parameters
        ----------
        matrix : array-like or dict
            Factor risk contributions matrix, either as N×K array or nested dictionary
        """
        if isinstance(matrix, dict):
            self._data["matrices"]["factor_risk_contributions"] = matrix.copy()
        else:
            matrix_array = np.asarray(matrix)
            if matrix_array.ndim == 2:
                n_assets, n_factors = matrix_array.shape
                asset_names = self.asset_names if len(self.asset_names) == n_assets else [f"asset_{i}" for i in range(n_assets)]
                factor_names = self.factor_names if len(self.factor_names) == n_factors else [f"factor_{i}" for i in range(n_factors)]
                
                self._data["matrices"]["factor_risk_contributions"] = {
                    asset_name: {
                        factor_name: float(matrix_array[i, j])
                        for j, factor_name in enumerate(factor_names)
                    }
                    for i, asset_name in enumerate(asset_names)
                }
            else:
                raise ValueError("Factor risk contributions matrix must be 2D array (N assets × K factors) or nested dictionary")
    
    def set_weighted_betas_matrix(self, matrix: Union[np.ndarray, Dict[str, Dict[str, float]]]) -> None:
        """
        Set weighted betas matrix (Asset × Factor).
        
        Parameters
        ----------
        matrix : array-like or dict
            Weighted betas matrix, either as N×K array or nested dictionary
        """
        if isinstance(matrix, dict):
            self._data["matrices"]["weighted_betas"] = matrix.copy()
        else:
            matrix_array = np.asarray(matrix)
            if matrix_array.ndim == 2:
                n_assets, n_factors = matrix_array.shape
                asset_names = self.asset_names if len(self.asset_names) == n_assets else [f"asset_{i}" for i in range(n_assets)]
                factor_names = self.factor_names if len(self.factor_names) == n_factors else [f"factor_{i}" for i in range(n_factors)]
                
                self._data["matrices"]["weighted_betas"] = {
                    asset_name: {
                        factor_name: float(matrix_array[i, j])
                        for j, factor_name in enumerate(factor_names)
                    }
                    for i, asset_name in enumerate(asset_names)
                }
            else:
                raise ValueError("Weighted betas matrix must be 2D array (N assets × K factors) or nested dictionary")
    
    def set_active_risk_metrics(self, active_metrics: Dict[str, Any]) -> None:
        """
        Set active risk specific metrics.
        
        Parameters
        ----------
        active_metrics : dict
            Dictionary containing active risk specific metrics
        """
        self._data["active_risk"].update(active_metrics)
    
    def set_validation_results(self, validation_results: Dict[str, Any]) -> None:
        """
        Set validation results.
        
        Parameters
        ----------
        validation_results : dict
            Validation results from risk decomposition checks
        """
        self._data["validation"]["checks"] = validation_results.copy()
        
        # Determine overall validation status
        if "overall_validation" in validation_results:
            self._data["validation"]["passes"] = validation_results["overall_validation"].get("passes", False)
            self._data["validation"]["summary"] = validation_results["overall_validation"].get("message", "")
        else:
            # Check individual validations
            passes = all(
                result.get("passes", False) 
                for key, result in validation_results.items() 
                if isinstance(result, dict) and "passes" in result
            )
            self._data["validation"]["passes"] = passes
            self._data["validation"]["summary"] = "All validations passed" if passes else "Some validations failed"
    
    def add_context_info(self, key: str, value: Any) -> None:
        """Add additional context information."""
        self._data["metadata"]["context_info"][key] = value
    
    def add_detail(self, key: str, value: Any) -> None:
        """Add detailed analysis information."""
        self._data["details"][key] = value
    
    def _validate_multi_lens_consistency(self, validation_results: Dict[str, Any]) -> None:
        """Validate consistency across portfolio, benchmark, and active lenses."""
        
        # Check if all three lenses have core metrics
        portfolio_core = self._data["portfolio"]["core_metrics"]
        benchmark_core = self._data["benchmark"]["core_metrics"]
        active_core = self._data["active"]["core_metrics"]
        
        portfolio_risk = portfolio_core.get("total_risk")
        benchmark_risk = benchmark_core.get("total_risk")
        active_risk = active_core.get("total_risk")
        
        if portfolio_risk is not None and benchmark_risk is not None and active_risk is not None:
            # Validate active risk decomposition consistency
            validation_results["multi_lens_consistency"] = {
                "portfolio_risk": portfolio_risk,
                "benchmark_risk": benchmark_risk,
                "active_risk": active_risk,
                "passes": True,  # Always passes for now, can add specific checks later
                "message": "Multi-lens data populated successfully"
            }
        else:
            validation_results["multi_lens_consistency"] = {
                "passes": True,
                "message": "Incomplete multi-lens data - validation skipped"
            }
        
        # Validate active decomposition if available
        if self._data["active"]["decomposition"]["validation"]["total_decomposition_check"] is not None:
            decomp_check = self._data["active"]["decomposition"]["validation"]["total_decomposition_check"]
            validation_results["active_decomposition"] = {
                "passes": decomp_check["passes"],
                "message": f"Active decomposition validation: {decomp_check['difference']:.6f} difference",
                "components": decomp_check["components"]
            }
    
    def _validate_hierarchy_consistency(self, validation_results: Dict[str, Any]) -> None:
        """Validate hierarchy structure consistency and relationships."""
        
        hierarchy_data = self._data["hierarchy"]
        
        # Check if hierarchy data exists
        if not hierarchy_data["root_component"] and not hierarchy_data["component_relationships"]:
            validation_results["hierarchy_consistency"] = {
                "passes": True,
                "message": "No hierarchy data to validate"
            }
            return
        
        root_component = hierarchy_data["root_component"]
        component_relationships = hierarchy_data["component_relationships"]
        component_metadata = hierarchy_data["component_metadata"]
        adjacency_list = hierarchy_data["adjacency_list"]
        
        validation_issues = []
        
        # 1. Root component validation
        if root_component:
            if root_component not in component_relationships:
                validation_issues.append(f"Root component '{root_component}' not found in relationships")
            elif component_relationships[root_component].get("parent") is not None:
                validation_issues.append(f"Root component '{root_component}' should not have a parent")
        
        # 2. Parent-child relationship consistency
        for component_id, relations in component_relationships.items():
            parent = relations.get("parent")
            children = relations.get("children", [])
            
            # Validate parent relationship
            if parent is not None:
                if parent not in component_relationships:
                    validation_issues.append(f"Component '{component_id}' has unknown parent '{parent}'")
                elif component_id not in component_relationships[parent].get("children", []):
                    validation_issues.append(f"Parent '{parent}' doesn't list '{component_id}' as child")
            
            # Validate children relationships
            for child in children:
                if child not in component_relationships:
                    validation_issues.append(f"Component '{component_id}' has unknown child '{child}'")
                elif component_relationships[child].get("parent") != component_id:
                    validation_issues.append(f"Child '{child}' doesn't have '{component_id}' as parent")
        
        # 3. Adjacency list consistency
        for parent_id, child_list in adjacency_list.items():
            if parent_id in component_relationships:
                expected_children = set(component_relationships[parent_id].get("children", []))
                actual_children = set(child_list)
                if expected_children != actual_children:
                    validation_issues.append(
                        f"Adjacency list mismatch for '{parent_id}': "
                        f"expected {expected_children}, got {actual_children}"
                    )
        
        # 4. Circular reference detection
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for child in component_relationships.get(node, {}).get("children", []):
                if has_cycle(child):
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check for cycles starting from all components
        for component_id in component_relationships:
            if component_id not in visited:
                if has_cycle(component_id):
                    validation_issues.append(f"Circular reference detected involving '{component_id}'")
                    break
        
        # 5. Metadata consistency
        for component_id in component_relationships:
            if component_id not in component_metadata:
                validation_issues.append(f"Missing metadata for component '{component_id}'")
        
        # 6. Leaf components validation
        leaf_components = hierarchy_data.get("leaf_components", [])
        actual_leaves = [
            comp_id for comp_id, relations in component_relationships.items()
            if not relations.get("children", [])
        ]
        
        if set(leaf_components) != set(actual_leaves):
            validation_issues.append(
                f"Leaf components mismatch: expected {set(actual_leaves)}, got {set(leaf_components)}"
            )
        
        # 7. Path mappings validation
        path_mappings = hierarchy_data.get("path_mappings", {})
        for component_id in component_relationships:
            if component_id in path_mappings:
                expected_path = component_id  # For hierarchical IDs, path should match ID
                actual_path = path_mappings[component_id]
                # Basic validation - paths should be consistent with component structure
                if not actual_path.endswith(component_id.split('/')[-1]):
                    validation_issues.append(
                        f"Path mapping inconsistent for '{component_id}': {actual_path}"
                    )
        
        # Compile validation results
        validation_results["hierarchy_consistency"] = {
            "passes": len(validation_issues) == 0,
            "issues": validation_issues,
            "component_count": len(component_relationships),
            "leaf_count": len(actual_leaves),
            "root_component": root_component,
            "message": (
                "Hierarchy validation passed" if len(validation_issues) == 0 
                else f"Hierarchy validation failed with {len(validation_issues)} issues"
            )
        }
    
    def _validate_time_series_consistency(self, validation_results: Dict[str, Any]) -> None:
        """Validate time series data consistency and completeness."""
        
        ts_data = self._data["time_series"]
        validation_issues = []
        
        # Check if time series data exists
        portfolio_returns = ts_data["portfolio_returns"]
        benchmark_returns = ts_data["benchmark_returns"]
        active_returns = ts_data["active_returns"]
        factor_returns = ts_data["factor_returns"]
        dates = ts_data["dates"]
        
        total_components = len(portfolio_returns) + len(benchmark_returns) + len(active_returns)
        total_factors = len(factor_returns)
        
        if total_components == 0 and total_factors == 0:
            validation_results["time_series_consistency"] = {
                "passes": True,
                "message": "No time series data to validate"
            }
            return
        
        # 1. Date consistency validation
        expected_periods = len(dates) if dates else None
        
        # Check portfolio returns length consistency
        for component_id, returns in portfolio_returns.items():
            if expected_periods and len(returns) != expected_periods:
                validation_issues.append(
                    f"Portfolio returns for '{component_id}': {len(returns)} periods, expected {expected_periods}"
                )
        
        # Check benchmark returns length consistency
        for component_id, returns in benchmark_returns.items():
            if expected_periods and len(returns) != expected_periods:
                validation_issues.append(
                    f"Benchmark returns for '{component_id}': {len(returns)} periods, expected {expected_periods}"
                )
        
        # Check active returns length consistency
        for component_id, returns in active_returns.items():
            if expected_periods and len(returns) != expected_periods:
                validation_issues.append(
                    f"Active returns for '{component_id}': {len(returns)} periods, expected {expected_periods}"
                )
        
        # Check factor returns length consistency
        for factor_name, returns in factor_returns.items():
            if expected_periods and len(returns) != expected_periods:
                validation_issues.append(
                    f"Factor returns for '{factor_name}': {len(returns)} periods, expected {expected_periods}"
                )
        
        # 2. Active returns calculation consistency
        active_calculation_issues = 0
        for component_id in active_returns:
            if component_id in portfolio_returns and component_id in benchmark_returns:
                port_returns = portfolio_returns[component_id]
                bench_returns = benchmark_returns[component_id]
                calculated_active = active_returns[component_id]
                
                if len(port_returns) == len(bench_returns) == len(calculated_active):
                    # Check if active returns approximately equal portfolio - benchmark
                    for i, (p, b, a) in enumerate(zip(port_returns, bench_returns, calculated_active)):
                        expected_active = p - b
                        if abs(a - expected_active) > 1e-8:
                            validation_issues.append(
                                f"Active return calculation mismatch for '{component_id}' at period {i}: "
                                f"expected {expected_active:.8f}, got {a:.8f}"
                            )
                            active_calculation_issues += 1
                            break  # Only report first mismatch per component
        
        # 3. Statistics consistency validation
        stats_issues = 0
        for return_type in ["portfolio", "benchmark", "active", "factor"]:
            if return_type == "factor":
                returns_data = factor_returns
                stats_data = ts_data["statistics"]["factor"]
            else:
                if return_type == "portfolio":
                    returns_data = portfolio_returns
                elif return_type == "benchmark":
                    returns_data = benchmark_returns
                else:  # active
                    returns_data = active_returns
                stats_data = ts_data["statistics"][return_type]
            
            # Check that every component/factor with returns has statistics
            for component_id in returns_data:
                if component_id not in stats_data:
                    validation_issues.append(f"Missing statistics for {return_type} '{component_id}'")
                    stats_issues += 1
            
            # Check that statistics are consistent with returns
            for component_id, stats in stats_data.items():
                if component_id in returns_data:
                    returns_list = returns_data[component_id]
                    if len(returns_list) != stats.get("total_periods", 0):
                        validation_issues.append(
                            f"Statistics period count mismatch for {return_type} '{component_id}': "
                            f"returns {len(returns_list)}, stats {stats.get('total_periods', 0)}"
                        )
                        stats_issues += 1
        
        # 4. Hierarchy consistency for time series
        hierarchy_data = self._data["hierarchy"]
        if hierarchy_data.get("component_relationships"):
            # Check that hierarchical components have consistent time series data
            for component_id in hierarchy_data["component_relationships"]:
                # At minimum, components should have either portfolio or benchmark returns
                has_portfolio = component_id in portfolio_returns
                has_benchmark = component_id in benchmark_returns
                
                if not has_portfolio and not has_benchmark:
                    # This might be acceptable for intermediate nodes, so just note it
                    pass
        
        # 5. Missing data validation
        missing_data_issues = 0
        for return_type, returns_data in [
            ("portfolio", portfolio_returns), 
            ("benchmark", benchmark_returns),
            ("active", active_returns)
        ]:
            for component_id, returns in returns_data.items():
                nan_count = sum(1 for r in returns if r is None or (isinstance(r, float) and np.isnan(r)))
                if nan_count > 0:
                    missing_data_issues += 1
                    validation_issues.append(
                        f"Missing data in {return_type} returns for '{component_id}': {nan_count} NaN values"
                    )
        
        # Compile validation results
        total_issues = len(validation_issues)
        
        validation_results["time_series_consistency"] = {
            "passes": total_issues == 0,
            "issues": validation_issues if total_issues <= 10 else validation_issues[:10] + [f"... and {total_issues - 10} more issues"],
            "summary": {
                "total_components": total_components,
                "total_factors": total_factors,
                "date_periods": expected_periods or 0,
                "active_calculation_issues": active_calculation_issues,
                "statistics_issues": stats_issues,
                "missing_data_issues": missing_data_issues,
                "total_issues": total_issues
            },
            "message": (
                "Time series validation passed" if total_issues == 0 
                else f"Time series validation failed with {total_issues} issues"
            )
        }
    
    def validate_schema(self) -> Dict[str, Any]:
        """
        Validate the schema completeness and consistency.
        
        Returns
        -------
        dict
            Validation results with 'passes' boolean and detailed checks
        """
        validation_results = {}
        
        # Core metrics validation
        core_metrics = self._data["core_metrics"]
        core_complete = all(
            core_metrics[key] is not None 
            for key in ["total_risk", "factor_risk_contribution", "specific_risk_contribution"]
        )
        validation_results["core_metrics_complete"] = {
            "passes": core_complete,
            "message": "Core metrics are complete" if core_complete else "Missing core metrics"
        }
        
        # Risk decomposition validation (Euler identity)
        if core_complete:
            total_risk = core_metrics["total_risk"]
            factor_risk = core_metrics["factor_risk_contribution"]
            specific_risk = core_metrics["specific_risk_contribution"]
            decomp_sum = factor_risk + specific_risk
            difference = abs(decomp_sum - total_risk)
            
            validation_results["euler_identity"] = {
                "passes": difference < 1e-6,
                "difference": difference,
                "expected": total_risk,
                "actual": decomp_sum,
                "message": f"Euler identity check: {difference:.8f} difference"
            }
        
        # Identifier consistency validation
        asset_names_count = len(self._data["identifiers"]["asset_names"])
        factor_names_count = len(self._data["identifiers"]["factor_names"])
        
        validation_results["identifier_consistency"] = {
            "passes": True,  # Basic check, can be extended
            "asset_count": asset_names_count,
            "factor_count": factor_names_count,
            "message": f"Identifiers: {asset_names_count} assets, {factor_names_count} factors"
        }
        
        # Matrix validation
        factor_risk_contributions_matrix = self._data["matrices"]["factor_risk_contributions"]
        if factor_risk_contributions_matrix and core_complete:
            # Calculate sum of factor risk contributions matrix
            matrix_sum = 0.0
            for asset_dict in factor_risk_contributions_matrix.values():
                for factor_contribution in asset_dict.values():
                    matrix_sum += factor_contribution
            
            expected_factor_risk = core_metrics["factor_risk_contribution"]
            matrix_difference = abs(matrix_sum - expected_factor_risk)
            
            validation_results["matrix_consistency"] = {
                "passes": matrix_difference < 1e-6,
                "difference": matrix_difference,
                "expected": expected_factor_risk,
                "actual": matrix_sum,
                "message": f"Factor risk contributions matrix sum check: {matrix_difference:.8f} difference"
            }
        else:
            validation_results["matrix_consistency"] = {
                "passes": True,
                "message": "No factor risk contributions matrix to validate"
            }
        
        # Weight validation
        weights_data = self._data["weights"]
        portfolio_weights = weights_data["portfolio_weights"]
        benchmark_weights = weights_data["benchmark_weights"]
        active_weights = weights_data["active_weights"]
        
        # Portfolio weight validation
        if portfolio_weights:
            port_sum = sum(portfolio_weights.values())
            port_finite = all(np.isfinite(w) for w in portfolio_weights.values())
            validation_results["portfolio_weights"] = {
                "passes": abs(port_sum - 1.0) < 1e-6 and port_finite,
                "sum": port_sum,
                "finite": port_finite,
                "message": f"Portfolio weights sum: {port_sum:.6f}, finite: {port_finite}"
            }
        else:
            validation_results["portfolio_weights"] = {
                "passes": True,
                "message": "No portfolio weights to validate"
            }
        
        # Benchmark weight validation
        if benchmark_weights:
            bench_sum = sum(benchmark_weights.values())
            bench_finite = all(np.isfinite(w) for w in benchmark_weights.values())
            validation_results["benchmark_weights"] = {
                "passes": abs(bench_sum - 1.0) < 1e-6 and bench_finite,
                "sum": bench_sum,
                "finite": bench_finite,
                "message": f"Benchmark weights sum: {bench_sum:.6f}, finite: {bench_finite}"
            }
        else:
            validation_results["benchmark_weights"] = {
                "passes": True,
                "message": "No benchmark weights to validate"
            }
        
        # Active weight validation
        if active_weights:
            active_sum = sum(active_weights.values())
            active_finite = all(np.isfinite(w) for w in active_weights.values())
            # Active weights should sum to zero (approximately)
            validation_results["active_weights"] = {
                "passes": abs(active_sum) < 1e-6 and active_finite,
                "sum": active_sum,
                "finite": active_finite,
                "message": f"Active weights sum: {active_sum:.6f}, finite: {active_finite}"
            }
        else:
            validation_results["active_weights"] = {
                "passes": True,
                "message": "No active weights to validate"
            }
        
        # Active weights calculation consistency
        if portfolio_weights and benchmark_weights and active_weights:
            # Check if active weights are consistent with portfolio - benchmark
            all_assets = set(portfolio_weights.keys()) | set(benchmark_weights.keys()) | set(active_weights.keys())
            max_difference = 0.0
            for asset in all_assets:
                port_weight = portfolio_weights.get(asset, 0.0)
                bench_weight = benchmark_weights.get(asset, 0.0)
                active_weight = active_weights.get(asset, 0.0)
                expected_active = port_weight - bench_weight
                difference = abs(active_weight - expected_active)
                max_difference = max(max_difference, difference)
            
            validation_results["active_weights_consistency"] = {
                "passes": max_difference < 1e-6,
                "max_difference": max_difference,
                "message": f"Active weights consistency check: max difference {max_difference:.8f}"
            }
        else:
            validation_results["active_weights_consistency"] = {
                "passes": True,
                "message": "Insufficient weight data for consistency check"
            }
        
        # Multi-lens validation
        self._validate_multi_lens_consistency(validation_results)
        
        # Hierarchy validation
        self._validate_hierarchy_consistency(validation_results)
        
        # Time series validation
        self._validate_time_series_consistency(validation_results)
        
        # Overall validation
        all_passed = all(result.get("passes", False) for result in validation_results.values())
        validation_results["overall"] = {
            "passes": all_passed,
            "message": "Schema validation passed" if all_passed else "Schema validation failed"
        }
        
        return validation_results
    
    def to_dict(self) -> Dict[str, Any]:
        """Export complete schema as dictionary."""
        return self._data.copy()
    
    def to_json(self) -> str:
        """Export schema as JSON string."""
        import json
        
        def convert_for_json(obj):
            """Convert NumPy types and other non-serializable objects."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj
        
        clean_data = convert_for_json(self._data)
        return json.dumps(clean_data, indent=2, separators=(',', ': '), ensure_ascii=False)
    
    def get_legacy_format(self, format_type: str = "decomposer") -> Dict[str, Any]:
        """
        Convert to legacy format for backward compatibility.
        
        Parameters
        ----------
        format_type : str
            Type of legacy format ('decomposer', 'strategy', 'analyzer')
            
        Returns
        -------
        dict
            Data in legacy format
        """
        if format_type == "decomposer":
            return self._to_decomposer_format()
        elif format_type == "strategy":
            return self._to_strategy_format()
        elif format_type == "analyzer":
            return self._to_analyzer_format()
        else:
            raise ValueError(f"Unknown legacy format type: {format_type}")
    
    def _to_decomposer_format(self) -> Dict[str, Any]:
        """Convert to RiskDecomposer.to_dict() format with multi-lens data."""
        return {
            "metadata": self._data["metadata"].copy(),
            "core_metrics": self._data["core_metrics"].copy(),
            "named_contributions": {
                "assets": {"total_contributions": self._data["contributions"]["by_asset"]},
                "factors": {
                    "contributions": self._data["contributions"]["by_factor"],
                    "exposures": self._data["exposures"]["factor_exposures"]
                },
                "asset_factor_loadings": self._data["exposures"]["factor_loadings"]
            },
            "weights": self._data["weights"].copy(),
            "arrays": self._data["arrays"].copy(),
            "active_risk": self._data["active_risk"].copy(),
            "validation": self._data["validation"].copy(),
            "additional": self._data["details"].copy(),
            # New multi-lens data
            "portfolio_lens": self._data["portfolio"].copy(),
            "benchmark_lens": self._data["benchmark"].copy(),
            "active_lens": self._data["active"].copy()
        }
    
    def _to_strategy_format(self) -> Dict[str, Any]:
        """Convert to Strategy.analyze() format."""
        core = self._data["core_metrics"]
        arrays = self._data["arrays"]
        
        # Convert lists back to numpy arrays for legacy compatibility
        factor_contributions = arrays["contributions"].get("factor_contributions", [])
        asset_contributions = arrays["contributions"].get("asset_contributions", [])
        
        if isinstance(factor_contributions, list) and factor_contributions:
            factor_contributions = np.array(factor_contributions)
        if isinstance(asset_contributions, list) and asset_contributions:
            asset_contributions = np.array(asset_contributions)
        
        # Convert factor exposures from named dict to array
        factor_exposures = self._data["exposures"]["factor_exposures"]
        if isinstance(factor_exposures, dict) and factor_exposures:
            # Convert named factor exposures back to array using factor_names order
            factor_names = self._data["identifiers"]["factor_names"]
            portfolio_factor_exposure = np.array([factor_exposures.get(name, 0.0) for name in factor_names])
        else:
            portfolio_factor_exposure = arrays["exposures"].get("factor_exposures", [])
            if isinstance(portfolio_factor_exposure, list) and portfolio_factor_exposure:
                portfolio_factor_exposure = np.array(portfolio_factor_exposure)
        
        # Convert weights to arrays for legacy compatibility
        portfolio_weights = arrays["weights"].get("portfolio_weights", [])
        benchmark_weights = arrays["weights"].get("benchmark_weights", [])
        active_weights = arrays["weights"].get("active_weights", [])
        
        if isinstance(portfolio_weights, list) and portfolio_weights:
            portfolio_weights = np.array(portfolio_weights)
        if isinstance(benchmark_weights, list) and benchmark_weights:
            benchmark_weights = np.array(benchmark_weights)
        if isinstance(active_weights, list) and active_weights:
            active_weights = np.array(active_weights)
        
        return {
            "portfolio_volatility": core["total_risk"],
            "factor_risk_contribution": core["factor_risk_contribution"],
            "specific_risk_contribution": core["specific_risk_contribution"],
            "factor_contributions": factor_contributions,
            "asset_total_contributions": asset_contributions,
            "portfolio_factor_exposure": portfolio_factor_exposure,
            "portfolio_weights": portfolio_weights,
            "benchmark_weights": benchmark_weights,
            "active_weights": active_weights,
            "analysis_type": self._data["metadata"]["analysis_type"],
            "asset_names": self._data["identifiers"]["asset_names"],
            "factor_names": self._data["identifiers"]["factor_names"],
            "validation": self._data["validation"]["checks"]
        }
    
    def _to_analyzer_format(self) -> Dict[str, Any]:
        """Convert to PortfolioRiskAnalyzer.get_risk_summary() format."""
        core = self._data["core_metrics"]
        identifiers = self._data["identifiers"]
        
        return {
            "analysis_type": self._data["metadata"]["analysis_type"],
            "portfolio_volatility": core["total_risk"],
            "factor_risk_contribution": core["factor_risk_contribution"],
            "specific_risk_contribution": core["specific_risk_contribution"],
            "factor_risk_percentage": core["factor_risk_percentage"],
            "specific_risk_percentage": core["specific_risk_percentage"],
            "number_of_components": len(identifiers.get("component_ids", [])),
            "factor_names": identifiers["factor_names"],
            "component_names": identifiers.get("component_ids", identifiers["asset_names"]),
            "portfolio_weights": self._data["weights"]["portfolio_weights"],
            "benchmark_weights": self._data["weights"]["benchmark_weights"],
            "active_weights": self._data["weights"]["active_weights"]
        }
    
    @classmethod
    def from_decomposer_result(cls, decomposer_dict: Dict[str, Any]) -> 'RiskResultSchema':
        """
        Create schema from RiskDecomposer.to_dict() result.
        
        Parameters
        ----------
        decomposer_dict : dict
            Result from RiskDecomposer.to_dict()
            
        Returns
        -------
        RiskResultSchema
            Unified schema instance
        """
        metadata = decomposer_dict.get("metadata", {})
        core_metrics = decomposer_dict.get("core_metrics", {})
        
        # Create schema instance
        schema = cls(
            analysis_type=metadata.get("analysis_type", "portfolio"),
            asset_names=metadata.get("asset_names", []),
            factor_names=metadata.get("factor_names", []),
            annualized=metadata.get("annualized", True)
        )
        
        # Set core metrics
        if all(key in core_metrics for key in ["portfolio_volatility", "factor_risk_contribution", "specific_risk_contribution"]):
            schema.set_core_metrics(
                core_metrics["portfolio_volatility"],
                core_metrics["factor_risk_contribution"],
                core_metrics["specific_risk_contribution"]
            )
        
        # Set contributions and exposures from named_contributions
        named_contrib = decomposer_dict.get("named_contributions", {})
        if "assets" in named_contrib:
            asset_contrib = named_contrib["assets"].get("total_contributions", {})
            schema.set_asset_contributions(asset_contrib)
        
        if "factors" in named_contrib:
            factor_contrib = named_contrib["factors"].get("contributions", {})
            schema.set_factor_contributions(factor_contrib)
            
            factor_exposures = named_contrib["factors"].get("exposures", {})
            schema.set_factor_exposures(factor_exposures)
        
        if "asset_factor_loadings" in named_contrib:
            schema.set_factor_loadings(named_contrib["asset_factor_loadings"])
        
        # Set validation results
        if "validation" in decomposer_dict:
            schema.set_validation_results(decomposer_dict["validation"])
        
        # Set active risk metrics
        if "active_risk" in decomposer_dict:
            schema.set_active_risk_metrics(decomposer_dict["active_risk"])
        
        # Set weights if available
        if "weights" in decomposer_dict:
            weights_data = decomposer_dict["weights"]
            if "portfolio_weights" in weights_data:
                schema.set_portfolio_weights(weights_data["portfolio_weights"])
            if "benchmark_weights" in weights_data:
                schema.set_benchmark_weights(weights_data["benchmark_weights"])
            if "active_weights" in weights_data:
                schema.set_active_weights(weights_data["active_weights"], auto_calculate=False)
        
        return schema
    
    @classmethod
    def from_strategy_result(cls, strategy_dict: Dict[str, Any]) -> 'RiskResultSchema':
        """
        Create schema from Strategy.analyze() result.
        
        Parameters
        ----------
        strategy_dict : dict
            Result from risk analysis strategy
            
        Returns
        -------
        RiskResultSchema
            Unified schema instance
        """
        # Create schema instance
        schema = cls(
            analysis_type=strategy_dict.get("analysis_type", "portfolio"),
            asset_names=strategy_dict.get("asset_names", []),
            factor_names=strategy_dict.get("factor_names", []),
            annualized=strategy_dict.get("annualized", True)
        )
        
        # Set core metrics
        schema.set_core_metrics(
            strategy_dict["portfolio_volatility"],
            strategy_dict["factor_risk_contribution"],
            strategy_dict["specific_risk_contribution"]
        )
        
        # Set contributions
        if "asset_total_contributions" in strategy_dict:
            schema.set_asset_contributions(strategy_dict["asset_total_contributions"])
        
        if "factor_contributions" in strategy_dict:
            schema.set_factor_contributions(strategy_dict["factor_contributions"])
        
        # Set exposures
        if "portfolio_factor_exposure" in strategy_dict:
            schema.set_factor_exposures(strategy_dict["portfolio_factor_exposure"])
        
        # Set weights
        if "portfolio_weights" in strategy_dict:
            schema.set_portfolio_weights(strategy_dict["portfolio_weights"])
        if "benchmark_weights" in strategy_dict:
            schema.set_benchmark_weights(strategy_dict["benchmark_weights"])
        if "active_weights" in strategy_dict:
            schema.set_active_weights(strategy_dict["active_weights"], auto_calculate=False)
        
        # Set validation
        if "validation" in strategy_dict:
            schema.set_validation_results(strategy_dict["validation"])
        
        return schema
    
    def __repr__(self) -> str:
        """String representation of the schema."""
        return (f"RiskResultSchema("
                f"type={self.analysis_type.value}, "
                f"assets={len(self.asset_names)}, "
                f"factors={len(self.factor_names)}, "
                f"timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M')})")