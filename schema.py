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
        """Create empty schema structure with all required sections."""
        return {
            "metadata": {
                "analysis_type": self.analysis_type.value,
                "timestamp": self.timestamp.isoformat(),
                "data_frequency": self.data_frequency,
                "annualized": self.annualized,
                "schema_version": "1.0",
                "context_info": {}
            },
            "identifiers": {
                "asset_names": self.asset_names.copy(),
                "factor_names": self.factor_names.copy(),
                "component_ids": self.component_ids.copy()
            },
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
            "weights": {
                "portfolio_weights": {},      # Named portfolio weights: {asset_name: weight}
                "benchmark_weights": {},      # Named benchmark weights: {asset_name: weight}
                "active_weights": {}          # Named active weights: {asset_name: weight}
            },
            "arrays": {
                "weights": {
                    "portfolio_weights": [],  # Array format for backward compatibility
                    "benchmark_weights": [],  # Array format for backward compatibility
                    "active_weights": []      # Array format for backward compatibility
                },
                "exposures": {},
                "contributions": {}
            },
            "matrices": {
                "factor_risk_contributions": {},  # Asset × Factor matrix: {asset_name: {factor_name: contribution}}
                "weighted_betas": {}              # Asset × Factor matrix: {asset_name: {factor_name: weighted_beta}}
            },
            "active_risk": {},              # Active risk specific metrics when applicable
            "validation": {
                "checks": {},
                "summary": "",
                "passes": True,
                "level": self.validation_level.value
            },
            "details": {}                   # Extended analysis details
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
        """Convert to RiskDecomposer.to_dict() format."""
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
            "additional": self._data["details"].copy()
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