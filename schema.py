"""
Unified Risk Result Schema
==========================

Standardized schema and validation for all risk analysis results across Spark.
This module provides a unified structure to eliminate overlaps and inconsistencies
between different risk analysis components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
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
                "schema_version": "3.0",  # Updated for complete hierarchical support
                "context_info": {}
            },
            "identifiers": {
                "asset_names": self.asset_names.copy(),
                "factor_names": self.factor_names.copy(),
                "component_ids": self.component_ids.copy()
            },
            
            # Complete hierarchical risk database - every component has full decomposition
            "hierarchical_risk_data": {
                # Structure: {component_id: {lens: {decomposer_results + validation}}}
                # Each component contains complete risk decomposition for all lenses
            },
            
            # Hierarchical matrices storage - matrices for every component and lens
            "hierarchical_matrices": {
                # Structure: {component_id: {lens: {matrix_type: matrix_data}}}
                # Contains all risk calculation matrices at component level
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
    
    # Hierarchical Risk Data Management Methods
    
    def set_component_full_decomposition(
        self,
        component_id: str,
        lens: str,
        decomposer_result: Dict[str, Any]
    ) -> None:
        """
        Set complete risk decomposition results for a specific component and lens.
        
        Parameters
        ----------
        component_id : str
            Component identifier (node or leaf)
        lens : str  
            Lens type ('portfolio', 'benchmark', or 'active')
        decomposer_result : dict
            Complete decomposer results dictionary containing:
            - total_risk, factor_risk_contribution, specific_risk_contribution
            - factor_contributions, asset_contributions  
            - factor_loadings_matrix, weighted_betas, risk_contribution_matrix
            - covariance_matrix, correlation_matrix, validation results
        """
        if lens not in ['portfolio', 'benchmark', 'active']:
            raise ValueError("lens must be 'portfolio', 'benchmark', or 'active'")
        
        # Initialize component if not exists
        if component_id not in self._data["hierarchical_risk_data"]:
            self._data["hierarchical_risk_data"][component_id] = {}
        
        # Set the complete decomposition data
        self._data["hierarchical_risk_data"][component_id][lens] = {
            "decomposer_results": {
                "total_risk": decomposer_result.get("total_risk", 0.0),
                "factor_risk_contribution": decomposer_result.get("factor_risk_contribution", 0.0),  
                "specific_risk_contribution": decomposer_result.get("specific_risk_contribution", 0.0),
                "factor_risk_percentage": decomposer_result.get("factor_risk_percentage", 0.0),
                "specific_risk_percentage": decomposer_result.get("specific_risk_percentage", 0.0),
                "factor_contributions": decomposer_result.get("factor_contributions", {}),
                "asset_contributions": decomposer_result.get("asset_contributions", {}),
                "factor_loadings_matrix": decomposer_result.get("factor_loadings_matrix", {}),
                "weighted_betas": decomposer_result.get("weighted_betas", {}),
                "risk_contribution_matrix": decomposer_result.get("risk_contribution_matrix", {}),
                "covariance_matrix": decomposer_result.get("covariance_matrix", []),
                "correlation_matrix": decomposer_result.get("correlation_matrix", [])
            },
            "validation": {
                "euler_identity_check": decomposer_result.get("euler_identity_check", True),
                "asset_sum_check": decomposer_result.get("asset_sum_check", True),
                "factor_sum_check": decomposer_result.get("factor_sum_check", True),
                "validation_summary": decomposer_result.get("validation_summary", ""),
                "validation_details": decomposer_result.get("validation_details", {})
            }
        }
        
        # Add active-specific data for active lens
        if lens == "active":
            allocation_selection = decomposer_result.get("allocation_selection", {})
            self._data["hierarchical_risk_data"][component_id][lens]["allocation_selection"] = {
                "allocation_factor_contributions": allocation_selection.get("allocation_factor_contributions", {}),
                "selection_factor_contributions": allocation_selection.get("selection_factor_contributions", {}),
                "interaction_contributions": allocation_selection.get("interaction_contributions", {}),
                "allocation_total": allocation_selection.get("allocation_total", 0.0),
                "selection_total": allocation_selection.get("selection_total", 0.0),
                "interaction_total": allocation_selection.get("interaction_total", 0.0)
            }
    
    def set_component_matrices(
        self,
        component_id: str,
        lens: str,
        matrices_dict: Dict[str, Any]
    ) -> None:
        """
        Set risk calculation matrices for a specific component and lens.
        
        Parameters
        ----------
        component_id : str
            Component identifier
        lens : str
            Lens type ('portfolio', 'benchmark', or 'active')  
        matrices_dict : dict
            Dictionary containing matrices:
            - beta_matrix, weighted_beta_matrix
            - factor_covariance, residual_covariance, total_covariance
            - descendant_leaves list for reference
        """
        if lens not in ['portfolio', 'benchmark', 'active']:
            raise ValueError("lens must be 'portfolio', 'benchmark', or 'active'")
        
        # Initialize component if not exists
        if component_id not in self._data["hierarchical_matrices"]:
            self._data["hierarchical_matrices"][component_id] = {}
        
        # Convert numpy arrays to lists for JSON serialization
        matrix_data = {}
        for key, value in matrices_dict.items():
            if hasattr(value, 'tolist'):  # numpy array
                matrix_data[key] = value.tolist()
            elif isinstance(value, list):
                matrix_data[key] = value
            else:
                matrix_data[key] = value
        
        self._data["hierarchical_matrices"][component_id][lens] = matrix_data
    
    def get_component_decomposition(
        self, 
        component_id: str,
        lens: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get complete risk decomposition for a specific component and lens.
        
        Parameters
        ----------
        component_id : str
            Component identifier
        lens : str
            Lens type ('portfolio', 'benchmark', or 'active')
            
        Returns
        -------
        dict or None
            Complete decomposition data or None if not found
        """
        hierarchical_data = self._data.get("hierarchical_risk_data", {})
        component_data = hierarchical_data.get(component_id, {})
        return component_data.get(lens)
    
    def get_component_matrices(
        self,
        component_id: str,
        lens: str,
        matrix_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get matrices for a specific component and lens.
        
        Parameters
        ----------
        component_id : str
            Component identifier
        lens : str
            Lens type ('portfolio', 'benchmark', or 'active')
        matrix_type : str, optional
            Specific matrix type to retrieve. If None, returns all matrices.
            
        Returns
        -------
        dict or None
            Matrix data or None if not found
        """
        matrices_data = self._data.get("hierarchical_matrices", {})
        component_matrices = matrices_data.get(component_id, {})
        lens_matrices = component_matrices.get(lens, {})
        
        if matrix_type:
            return lens_matrices.get(matrix_type)
        return lens_matrices
    
    def get_component_factor_data(
        self,
        component_id: str,
        lens: str
    ) -> Dict[str, Any]:
        """
        Get factor-specific data for a component and lens.
        
        Parameters
        ----------
        component_id : str
            Component identifier
        lens : str
            Lens type ('portfolio', 'benchmark', or 'active')
            
        Returns
        -------
        dict
            Dictionary with factor contributions, exposures, and loadings
        """
        decomposition = self.get_component_decomposition(component_id, lens)
        if not decomposition:
            return {}
        
        decomposer_results = decomposition.get("decomposer_results", {})
        return {
            "factor_contributions": decomposer_results.get("factor_contributions", {}),
            "factor_loadings_matrix": decomposer_results.get("factor_loadings_matrix", {}),
            "weighted_betas": decomposer_results.get("weighted_betas", {})
        }
    
    # Hierarchical Navigation Methods
    
    def get_available_components(self) -> List[str]:
        """Get list of all components with hierarchical risk data."""
        return list(self._data.get("hierarchical_risk_data", {}).keys())
    
    def get_hierarchical_risk_data(self) -> Dict[str, Dict[str, Any]]:
        """Get all hierarchical risk data."""
        return self._data.get("hierarchical_risk_data", {})
    
    def set_hierarchical_risk_data(self, hierarchical_data: Dict[str, Dict[str, Any]]) -> None:
        """Set hierarchical risk data."""
        self._data["hierarchical_risk_data"] = hierarchical_data
    
    def get_component_children(self, component_id: str) -> List[str]:
        """Get direct children of a component from hierarchy."""
        hierarchy = self._data.get("hierarchy", {})
        adjacency_list = hierarchy.get("adjacency_list", {})
        return adjacency_list.get(component_id, [])
    
    def get_component_parent(self, component_id: str) -> Optional[str]:
        """Get parent of a component from hierarchy."""
        hierarchy = self._data.get("hierarchy", {})
        component_relationships = hierarchy.get("component_relationships", {})
        component_info = component_relationships.get(component_id, {})
        return component_info.get("parent")
    
    def get_component_descendants(self, component_id: str) -> List[str]:
        """Get all descendant components (recursive children)."""
        descendants = []
        children = self.get_component_children(component_id)
        
        for child in children:
            descendants.append(child)
            # Recursively get grandchildren
            descendants.extend(self.get_component_descendants(child))
        
        return descendants
    
    def get_component_hierarchy_path(self, component_id: str) -> List[str]:
        """Get hierarchy path from root to component."""
        path = []
        current = component_id
        
        while current:
            path.append(current)
            current = self.get_component_parent(current)
        
        return list(reversed(path))  # Root to component order
    
    def can_drill_down(self, component_id: str) -> bool:
        """Check if component has children to drill down into."""
        return len(self.get_component_children(component_id)) > 0
    
    def can_drill_up(self, component_id: str) -> bool:
        """Check if component has parent to drill up to."""
        return self.get_component_parent(component_id) is not None
    
    # Bulk Hierarchical Storage Methods
    
    def populate_hierarchical_risk_data_from_visitor(
        self, 
        visitor: 'FactorRiskDecompositionVisitor',
        component_ids: Optional[List[str]] = None,
        lenses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Populate hierarchical risk data from visitor results for all components.
        
        This method extracts risk decomposition results from a visitor that has 
        traversed the portfolio graph and stores them in the hierarchical_risk_data
        section for easy retrieval.
        
        Parameters
        ----------
        visitor : FactorRiskDecompositionVisitor
            Visitor instance that has completed graph traversal
        component_ids : list of str, optional
            Specific components to extract (defaults to all processed components)
        lenses : list of str, optional
            Specific lenses to extract (defaults to ['portfolio', 'benchmark', 'active'])
            
        Returns
        -------
        dict
            Summary of populated data with counts and any errors
        """
        if lenses is None:
            lenses = ['portfolio', 'benchmark', 'active']
        
        summary = {
            'components_processed': 0,
            'lenses_populated': {},
            'errors': [],
            'success': True
        }
        
        # Get component IDs from visitor if not specified
        if component_ids is None:
            component_ids = []
            if hasattr(visitor, '_processed_components'):
                component_ids = list(visitor._processed_components)
            elif hasattr(visitor, 'metric_store') and visitor.metric_store:
                # Get all components that have hierarchical model context
                for comp_id in visitor.metric_store._metrics:
                    if visitor.metric_store.get_metric(comp_id, 'hierarchical_model_context'):
                        component_ids.append(comp_id)
        
        if not component_ids:
            summary['success'] = False
            summary['errors'].append("No component IDs found in visitor")
            return summary
        
        # Extract risk data for each component and lens
        for component_id in component_ids:
            try:
                component_data_found = False
                
                # Get hierarchical model context from visitor's metric store
                if hasattr(visitor, 'metric_store') and visitor.metric_store:
                    context_metric = visitor.metric_store.get_metric(component_id, 'hierarchical_model_context')
                    
                    if context_metric and hasattr(context_metric, 'value'):
                        hierarchical_context = context_metric.value()
                        
                        for lens in lenses:
                            decomposer = None
                            
                            # Get appropriate decomposer based on lens
                            if lens == 'portfolio' and hasattr(hierarchical_context, 'portfolio_decomposer'):
                                decomposer = hierarchical_context.portfolio_decomposer
                            elif lens == 'benchmark' and hasattr(hierarchical_context, 'benchmark_decomposer'):
                                decomposer = hierarchical_context.benchmark_decomposer
                            elif lens == 'active' and hasattr(hierarchical_context, 'active_decomposer'):
                                decomposer = hierarchical_context.active_decomposer
                            
                            if decomposer:
                                # Extract decomposer results
                                decomposer_result = {
                                    'total_risk': decomposer.portfolio_volatility,
                                    'factor_risk_contribution': decomposer.factor_risk_contribution,
                                    'specific_risk_contribution': decomposer.specific_risk_contribution,
                                    'factor_risk_percentage': (decomposer.factor_risk_contribution / decomposer.portfolio_volatility * 100) if decomposer.portfolio_volatility > 0 else 0.0,
                                    'specific_risk_percentage': (decomposer.specific_risk_contribution / decomposer.portfolio_volatility * 100) if decomposer.portfolio_volatility > 0 else 0.0,
                                    'factor_contributions': decomposer.factor_contributions if hasattr(decomposer, 'factor_contributions') else {},
                                    'asset_contributions': decomposer.asset_total_contributions if hasattr(decomposer, 'asset_total_contributions') else {},
                                    'weighted_betas': decomposer.weighted_betas if hasattr(decomposer, 'weighted_betas') else {},
                                }
                                
                                # Store using existing method
                                self.set_component_full_decomposition(component_id, lens, decomposer_result)
                                component_data_found = True
                                
                                # Track lens population
                                if lens not in summary['lenses_populated']:
                                    summary['lenses_populated'][lens] = 0
                                summary['lenses_populated'][lens] += 1
                
                if component_data_found:
                    summary['components_processed'] += 1
                else:
                    summary['errors'].append(f"No risk data found for component '{component_id}'")
                    
            except Exception as e:
                summary['errors'].append(f"Error processing component '{component_id}': {str(e)}")
                summary['success'] = False
        
        return summary
    
    def get_all_component_risk_results(
        self,
        lens: Optional[str] = None,
        include_matrices: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get risk results for all components in hierarchical structure.
        
        Parameters
        ----------
        lens : str, optional
            Specific lens to retrieve ('portfolio', 'benchmark', 'active').
            If None, returns all lenses
        include_matrices : bool, default False
            Whether to include matrix data in results
            
        Returns
        -------
        dict
            Component risk results: {component_id: {lens: risk_data}}
        """
        hierarchical_data = self._data.get("hierarchical_risk_data", {})
        results = {}
        
        for component_id, component_data in hierarchical_data.items():
            results[component_id] = {}
            
            if lens is None:
                # Return all lenses
                for lens_type, lens_data in component_data.items():
                    results[component_id][lens_type] = lens_data.copy()
            else:
                # Return specific lens
                if lens in component_data:
                    results[component_id][lens] = component_data[lens].copy()
            
            # Add matrices if requested
            if include_matrices:
                matrices = self.get_component_matrices(component_id, lens or 'portfolio')
                if matrices:
                    if lens:
                        if lens in results[component_id]:
                            results[component_id][lens]['matrices'] = matrices
                    else:
                        for lens_type in results[component_id]:
                            matrices_data = self.get_component_matrices(component_id, lens_type)
                            if matrices_data:
                                results[component_id][lens_type]['matrices'] = matrices_data
        
        return results
    
    def get_component_risk_summary(self, component_id: str) -> Dict[str, Any]:
        """
        Get focused risk summary for a specific component.
        
        Parameters
        ----------
        component_id : str
            Component identifier
            
        Returns
        -------
        dict
            Focused component risk summary with all lenses and navigation info
        """
        summary = {
            'component_id': component_id,
            'exists': component_id in self._data.get("hierarchical_risk_data", {}),
            'risk_data': {},
            'navigation': {},
            'metadata': {}
        }
        
        if summary['exists']:
            # Get risk data for all lenses
            summary['risk_data'] = self.get_component_decomposition(component_id, 'portfolio') or {}
            
            # Get navigation info
            summary['navigation'] = {
                'parent': self.get_component_parent(component_id),
                'children': self.get_component_children(component_id),
                'descendants': self.get_component_descendants(component_id),
                'hierarchy_path': self.get_component_hierarchy_path(component_id),
                'can_drill_up': self.can_drill_up(component_id),
                'can_drill_down': self.can_drill_down(component_id)
            }
            
            # Get metadata
            hierarchy = self._data.get("hierarchy", {})
            component_metadata = hierarchy.get("component_metadata", {})
            summary['metadata'] = component_metadata.get(component_id, {})
        
        return summary
    
    def validate_hierarchical_completeness(
        self,
        required_lenses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate that all components have complete hierarchical risk data.
        
        Parameters
        ----------
        required_lenses : list of str, optional
            Lenses that must be present for each component.
            Defaults to ['portfolio', 'benchmark', 'active']
            
        Returns
        -------
        dict
            Validation results with completeness analysis
        """
        if required_lenses is None:
            required_lenses = ['portfolio', 'benchmark', 'active']
        
        hierarchical_data = self._data.get("hierarchical_risk_data", {})
        hierarchy = self._data.get("hierarchy", {})
        component_relationships = hierarchy.get("component_relationships", {})
        
        validation_results = {
            'complete': True,
            'total_components': len(component_relationships),
            'components_with_data': len(hierarchical_data),
            'missing_components': [],
            'incomplete_components': {},
            'lens_coverage': {},
            'summary': {}
        }
        
        # Check which components are missing entirely
        for component_id in component_relationships:
            if component_id not in hierarchical_data:
                validation_results['missing_components'].append(component_id)
                validation_results['complete'] = False
        
        # Check lens completeness for components that have data
        for lens in required_lenses:
            validation_results['lens_coverage'][lens] = 0
            
        for component_id, component_data in hierarchical_data.items():
            missing_lenses = []
            for lens in required_lenses:
                if lens in component_data:
                    validation_results['lens_coverage'][lens] += 1
                else:
                    missing_lenses.append(lens)
            
            if missing_lenses:
                validation_results['incomplete_components'][component_id] = missing_lenses
                validation_results['complete'] = False
        
        # Generate summary
        validation_results['summary'] = {
            'completeness_percentage': (validation_results['components_with_data'] / max(validation_results['total_components'], 1)) * 100,
            'missing_count': len(validation_results['missing_components']),
            'incomplete_count': len(validation_results['incomplete_components']),
            'fully_complete_count': validation_results['components_with_data'] - len(validation_results['incomplete_components'])
        }
        
        return validation_results
    
    
    
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
    
    def _validate_full_hierarchy_consistency(self, validation_results: Dict[str, Any]) -> None:
        """Validate hierarchical risk data consistency and completeness."""
        
        hierarchical_data = self._data.get("hierarchical_risk_data", {})
        hierarchical_matrices = self._data.get("hierarchical_matrices", {})
        validation_issues = []
        
        if not hierarchical_data:
            validation_results["hierarchical_risk_data"] = {
                "passes": True,
                "message": "No hierarchical risk data to validate"
            }
            return
        
        # 1. Validate that all components have data for all lenses
        all_components = set(hierarchical_data.keys())
        expected_lenses = {'portfolio', 'benchmark', 'active'}
        
        for component_id in all_components:
            component_data = hierarchical_data[component_id]
            available_lenses = set(component_data.keys())
            missing_lenses = expected_lenses - available_lenses
            
            if missing_lenses:
                validation_issues.append(
                    f"Component '{component_id}' missing lens data: {missing_lenses}"
                )
        
        # 2. Validate decomposition consistency within each component
        decomposition_issues = 0
        for component_id, component_data in hierarchical_data.items():
            for lens, lens_data in component_data.items():
                decomposer_results = lens_data.get("decomposer_results", {})
                
                # Check Euler identity for each component
                total_risk = decomposer_results.get("total_risk", 0.0)
                factor_risk = decomposer_results.get("factor_risk_contribution", 0.0)
                specific_risk = decomposer_results.get("specific_risk_contribution", 0.0)
                
                if total_risk > 0:
                    decomp_sum = factor_risk + specific_risk
                    difference = abs(decomp_sum - total_risk)
                    
                    if difference > 1e-6:  # Tolerance for numerical precision
                        validation_issues.append(
                            f"Euler identity violation in '{component_id}' {lens}: "
                            f"difference {difference:.8f}"
                        )
                        decomposition_issues += 1
        
        # 3. Validate hierarchical aggregation consistency
        hierarchy = self._data.get("hierarchy", {})
        component_relationships = hierarchy.get("component_relationships", {})
        aggregation_issues = 0
        
        for component_id, relationships in component_relationships.items():
            children = relationships.get("children", [])
            if not children:  # Skip leaf components
                continue
            
            # Check if parent risk approximately equals sum of children risks
            for lens in expected_lenses:
                parent_data = hierarchical_data.get(component_id, {}).get(lens, {})
                parent_results = parent_data.get("decomposer_results", {})
                parent_total_risk = parent_results.get("total_risk", 0.0)
                
                if parent_total_risk == 0.0:
                    continue  # Skip if no parent data
                
                children_total_risk = 0.0
                children_with_data = 0
                
                for child_id in children:
                    child_data = hierarchical_data.get(child_id, {}).get(lens, {})
                    child_results = child_data.get("decomposer_results", {})
                    child_risk = child_results.get("total_risk", 0.0)
                    
                    if child_risk > 0:
                        children_total_risk += child_risk ** 2  # Risk aggregation via variance
                        children_with_data += 1
                
                if children_with_data > 0:
                    # Compare with parent (accounting for diversification)
                    children_aggregated_risk = np.sqrt(children_total_risk)
                    risk_ratio = children_aggregated_risk / parent_total_risk if parent_total_risk > 0 else 0
                    
                    # Allow reasonable diversification (children sum can be higher than parent)
                    if risk_ratio > 2.0 or risk_ratio < 0.5:  # Outside reasonable bounds
                        validation_issues.append(
                            f"Hierarchical risk aggregation issue in '{component_id}' {lens}: "
                            f"parent={parent_total_risk:.6f}, children_agg={children_aggregated_risk:.6f}, "
                            f"ratio={risk_ratio:.2f}"
                        )
                        aggregation_issues += 1
        
        # 4. Validate matrix consistency
        matrix_issues = 0
        for component_id in all_components:
            component_matrices = hierarchical_matrices.get(component_id, {})
            component_risk_data = hierarchical_data.get(component_id, {})
            
            for lens in expected_lenses:
                lens_matrices = component_matrices.get(lens, {})
                lens_risk_data = component_risk_data.get(lens, {})
                
                if lens_matrices and lens_risk_data:
                    # Check matrix dimensions consistency
                    beta_matrix = lens_matrices.get("beta_matrix", [])
                    factor_contributions = lens_risk_data.get("decomposer_results", {}).get("factor_contributions", {})
                    
                    if beta_matrix and factor_contributions:
                        n_factors_matrix = len(beta_matrix[0]) if beta_matrix else 0
                        n_factors_contrib = len(factor_contributions)
                        
                        if n_factors_matrix != n_factors_contrib:
                            validation_issues.append(
                                f"Matrix dimension mismatch in '{component_id}' {lens}: "
                                f"beta_matrix factors={n_factors_matrix}, contributions factors={n_factors_contrib}"
                            )
                            matrix_issues += 1
        
        # Compile validation results
        total_issues = len(validation_issues)
        
        validation_results["hierarchical_risk_consistency"] = {
            "passes": total_issues == 0,
            "issues": validation_issues if total_issues <= 20 else validation_issues[:20] + [f"... and {total_issues - 20} more issues"],
            "summary": {
                "total_components": len(all_components),
                "decomposition_issues": decomposition_issues,
                "aggregation_issues": aggregation_issues,
                "matrix_issues": matrix_issues,
                "total_issues": total_issues
            },
            "message": (
                "Hierarchical risk validation passed" if total_issues == 0 
                else f"Hierarchical risk validation failed with {total_issues} issues"
            )
        }
    
    def validate_component_decomposition(self, component_id: str) -> Dict[str, Any]:
        """
        Validate risk decomposition for a specific component.
        
        Parameters
        ----------
        component_id : str
            Component identifier to validate
            
        Returns
        -------
        dict
            Validation results for the component
        """
        hierarchical_data = self._data.get("hierarchical_risk_data", {})
        component_data = hierarchical_data.get(component_id, {})
        
        if not component_data:
            return {
                "passes": False,
                "message": f"No hierarchical data found for component '{component_id}'"
            }
        
        validation_results = {}
        
        for lens, lens_data in component_data.items():
            decomposer_results = lens_data.get("decomposer_results", {})
            
            # Validate Euler identity
            total_risk = decomposer_results.get("total_risk", 0.0)
            factor_risk = decomposer_results.get("factor_risk_contribution", 0.0)
            specific_risk = decomposer_results.get("specific_risk_contribution", 0.0)
            
            decomp_sum = factor_risk + specific_risk
            difference = abs(decomp_sum - total_risk) if total_risk > 0 else 0
            
            validation_results[f"{lens}_euler_identity"] = {
                "passes": difference < 1e-6,
                "difference": difference,
                "expected": total_risk,
                "actual": decomp_sum,
                "message": f"{lens} Euler identity check: {difference:.8f} difference"
            }
            
            # Validate factor contributions sum
            factor_contributions = decomposer_results.get("factor_contributions", {})
            if factor_contributions:
                factor_sum = sum(abs(contrib) for contrib in factor_contributions.values())
                expected_factor = abs(factor_risk)
                factor_diff = abs(factor_sum - expected_factor)
                
                validation_results[f"{lens}_factor_contributions"] = {
                    "passes": factor_diff < 1e-6,
                    "difference": factor_diff,
                    "expected": expected_factor,
                    "actual": factor_sum,
                    "message": f"{lens} factor contributions check: {factor_diff:.8f} difference"
                }
        
        # Overall component validation
        all_passed = all(result.get("passes", False) for result in validation_results.values())
        validation_results["overall"] = {
            "passes": all_passed,
            "component_id": component_id,
            "message": f"Component validation {'passed' if all_passed else 'failed'}"
        }
        
        return validation_results
    
    def validate_hierarchical_aggregation(self) -> Dict[str, Any]:
        """
        Validate that parent component risks properly aggregate from children.
        
        Returns
        -------
        dict
            Hierarchical aggregation validation results
        """
        hierarchical_data = self._data.get("hierarchical_risk_data", {})
        hierarchy = self._data.get("hierarchy", {})
        component_relationships = hierarchy.get("component_relationships", {})
        
        validation_results = {}
        
        for parent_id, relationships in component_relationships.items():
            children = relationships.get("children", [])
            if not children:  # Skip leaf components
                continue
            
            parent_validation = {}
            
            for lens in ['portfolio', 'benchmark', 'active']:
                parent_data = hierarchical_data.get(parent_id, {}).get(lens, {})
                parent_results = parent_data.get("decomposer_results", {})
                parent_total_risk = parent_results.get("total_risk", 0.0)
                
                # Calculate expected risk from children
                children_risks = []
                for child_id in children:
                    child_data = hierarchical_data.get(child_id, {}).get(lens, {})
                    child_results = child_data.get("decomposer_results", {})
                    child_risk = child_results.get("total_risk", 0.0)
                    if child_risk > 0:
                        children_risks.append(child_risk)
                
                if children_risks and parent_total_risk > 0:
                    # Simple aggregation check (actual implementation may use correlation)
                    children_sum_of_squares = sum(risk ** 2 for risk in children_risks)
                    expected_parent_risk = np.sqrt(children_sum_of_squares)
                    
                    difference = abs(parent_total_risk - expected_parent_risk)
                    relative_diff = difference / parent_total_risk if parent_total_risk > 0 else 0
                    
                    parent_validation[f"{lens}_aggregation"] = {
                        "passes": relative_diff < 0.5,  # Allow 50% difference for diversification
                        "parent_risk": parent_total_risk,
                        "expected_risk": expected_parent_risk,
                        "difference": difference,
                        "relative_difference": relative_diff,
                        "children_count": len(children_risks),
                        "message": f"{lens} aggregation check: {relative_diff:.1%} relative difference"
                    }
            
            validation_results[parent_id] = parent_validation
        
        # Overall aggregation validation
        all_checks = []
        for parent_results in validation_results.values():
            all_checks.extend([check.get("passes", False) for check in parent_results.values()])
        
        overall_passed = all(all_checks) if all_checks else True
        validation_results["overall"] = {
            "passes": overall_passed,
            "parents_checked": len(validation_results),
            "total_checks": len(all_checks),
            "message": f"Hierarchical aggregation {'passed' if overall_passed else 'failed'}"
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
    
    
    # =====================================================================
    # CONSOLIDATED UI ACCESS METHODS - Single Source of Truth
    # =====================================================================
    
    def get_ui_metrics(self, component_id: str = "TOTAL", lens: str = "portfolio") -> Dict[str, float]:
        """
        Get UI-ready core metrics for a component and lens.
        
        Parameters
        ----------
        component_id : str, default "TOTAL"
            Component identifier
        lens : str, default "portfolio"
            Analysis lens (portfolio/benchmark/active)
            
        Returns
        -------
        Dict[str, float]
            Core metrics: total_risk, factor_risk_contribution, specific_risk_contribution, factor_risk_percentage
        """
        # Try lens-specific data first
        if lens in self._data:
            lens_data = self._data[lens].get("core_metrics", {})
            if any(v is not None for v in lens_data.values()):
                return {
                    "total_risk": lens_data.get("total_risk", 0.0) or 0.0,
                    "factor_risk_contribution": lens_data.get("factor_risk_contribution", 0.0) or 0.0,
                    "specific_risk_contribution": lens_data.get("specific_risk_contribution", 0.0) or 0.0,
                    "factor_risk_percentage": lens_data.get("factor_risk_percentage", 0.0) or 0.0
                }
        
        # Return empty metrics if no lens data available
        return {
            "total_risk": 0.0,
            "factor_risk_contribution": 0.0,
            "specific_risk_contribution": 0.0,
            "factor_risk_percentage": 0.0
        }
    
    def get_ui_contributions(self, component_id: str = "TOTAL", lens: str = "portfolio", contrib_type: str = "by_factor") -> Dict[str, float]:
        """
        Get UI-ready contributions for a component and lens.
        
        Parameters
        ----------
        component_id : str, default "TOTAL"
            Component identifier
        lens : str, default "portfolio"
            Analysis lens (portfolio/benchmark/active)
        contrib_type : str, default "by_factor"
            Type of contributions (by_factor, by_asset, by_component)
            
        Returns
        -------
        Dict[str, float]
            Named contributions
        """
        # Try lens-specific data first
        if lens in self._data:
            lens_contribs = self._data[lens].get("contributions", {}).get(contrib_type, {})
            if lens_contribs:
                return dict(lens_contribs)
        
        # Return empty if no lens data available
        return {}
    
    def get_ui_exposures(self, component_id: str = "TOTAL", lens: str = "portfolio") -> Dict[str, float]:
        """
        Get UI-ready factor exposures for a component and lens.
        
        Parameters
        ----------
        component_id : str, default "TOTAL"
            Component identifier
        lens : str, default "portfolio"
            Analysis lens (portfolio/benchmark/active)
            
        Returns
        -------
        Dict[str, float]
            Factor exposures
        """
        # Try lens-specific data first
        if lens in self._data:
            lens_exposures = self._data[lens].get("exposures", {}).get("factor_exposures", {})
            if lens_exposures:
                return dict(lens_exposures)
        
        # Return empty if no lens data available
        return {}
    
    def get_ui_weights(self, component_id: str = "TOTAL") -> Dict[str, Dict[str, float]]:
        """
        Get UI-ready weight data for a component.
        
        Parameters
        ----------
        component_id : str, default "TOTAL"
            Component identifier
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Weight data: {portfolio_weights: {...}, benchmark_weights: {...}, active_weights: {...}}
        """
        weights_data = self._data.get("weights", {})
        return {
            "portfolio_weights": dict(weights_data.get("portfolio_weights", {})),
            "benchmark_weights": dict(weights_data.get("benchmark_weights", {})),
            "active_weights": dict(weights_data.get("active_weights", {}))
        }
    
    def get_ui_matrices(self, component_id: str = "TOTAL", lens: str = "portfolio") -> Dict[str, Any]:
        """
        Get UI-ready matrix data for a component and lens.
        
        Parameters
        ----------
        component_id : str, default "TOTAL"
            Component identifier
        lens : str, default "portfolio"
            Analysis lens (portfolio/benchmark/active)
            
        Returns
        -------
        Dict[str, Any]
            Matrix data
        """
        # Try lens-specific matrices first
        if lens in self._data:
            lens_matrices = self._data[lens].get("matrices", {})
            if lens_matrices:
                return dict(lens_matrices)
        
        # Return empty if no lens data available
        return {}
    
    def to_ui_format(self, component_id: str = "TOTAL") -> Dict[str, Any]:
        """
        Convert schema to UI-ready format with all data accessible through consistent paths.
        
        Parameters
        ----------
        component_id : str, default "TOTAL"
            Component identifier for the conversion
            
        Returns
        -------
        Dict[str, Any]
            UI-ready data format matching streamlit-skeleton.md specification
        """
        ui_data = {
            "metadata": {
                "analysis_type": self.analysis_type.value,
                "timestamp": self.timestamp.isoformat(),
                "data_frequency": self.data_frequency,
                "annualized": self.annualized,
                "schema_version": "3.0"
            },
            
            # Core metrics by lens
            "core_metrics": self.get_ui_metrics(component_id, "portfolio"),
            
            # Lens-specific sections
            "portfolio": {
                "core_metrics": self.get_ui_metrics(component_id, "portfolio"),
                "contributions": {
                    "by_factor": self.get_ui_contributions(component_id, "portfolio", "by_factor"),
                    "by_asset": self.get_ui_contributions(component_id, "portfolio", "by_asset"),
                    "by_component": self.get_ui_contributions(component_id, "portfolio", "by_component")
                },
                "exposures": {
                    "factor_exposures": self.get_ui_exposures(component_id, "portfolio")
                },
                "matrices": self.get_ui_matrices(component_id, "portfolio")
            },
            
            "benchmark": {
                "core_metrics": self.get_ui_metrics(component_id, "benchmark"),
                "contributions": {
                    "by_factor": self.get_ui_contributions(component_id, "benchmark", "by_factor"),
                    "by_asset": self.get_ui_contributions(component_id, "benchmark", "by_asset"),
                    "by_component": self.get_ui_contributions(component_id, "benchmark", "by_component")
                },
                "exposures": {
                    "factor_exposures": self.get_ui_exposures(component_id, "benchmark")
                },
                "matrices": self.get_ui_matrices(component_id, "benchmark")
            },
            
            "active": {
                "core_metrics": self.get_ui_metrics(component_id, "active"),
                "contributions": {
                    "by_factor": self.get_ui_contributions(component_id, "active", "by_factor"),
                    "by_asset": self.get_ui_contributions(component_id, "active", "by_asset"),
                    "by_component": self.get_ui_contributions(component_id, "active", "by_component")
                },
                "exposures": {
                    "factor_exposures": self.get_ui_exposures(component_id, "active")
                },
                "matrices": self.get_ui_matrices(component_id, "active"),
                "decomposition": self._data.get("active", {}).get("decomposition", {})
            },
            
            # Global sections
            "weights": self.get_ui_weights(component_id),
            "hierarchy": self._data.get("hierarchy", {}),
            "time_series": self._data.get("time_series", {}),
            "validation": self._data.get("validation", {}),
            "identifiers": {
                "asset_names": self.asset_names,
                "factor_names": self.factor_names,
                "component_ids": self.component_ids
            }
        }
        
        return ui_data
    
    def validate_comprehensive(self) -> Dict[str, Any]:
        """
        Comprehensive validation of schema data.
        
        Returns
        -------
        Dict[str, Any]
            Validation results with detailed checks
        """
        validation_results = {
            "passes": True,
            "checks": {},
            "warnings": [],
            "errors": []
        }
        
        # Core metrics validation
        core_metrics = self._data.get("core_metrics", {})
        total_risk = core_metrics.get("total_risk")
        factor_contrib = core_metrics.get("factor_risk_contribution")
        specific_contrib = core_metrics.get("specific_risk_contribution")
        
        if total_risk is not None and factor_contrib is not None and specific_contrib is not None:
            sum_check = abs((factor_contrib + specific_contrib) - total_risk)
            validation_results["checks"]["euler_identity"] = sum_check < 1e-10
            if sum_check >= 1e-10:
                validation_results["errors"].append(f"Euler identity failed: sum difference = {sum_check}")
                validation_results["passes"] = False
        
        # Weights validation
        weights = self._data.get("weights", {})
        portfolio_weights = weights.get("portfolio_weights", {})
        benchmark_weights = weights.get("benchmark_weights", {})
        active_weights = weights.get("active_weights", {})
        
        if portfolio_weights:
            port_sum = sum(portfolio_weights.values())
            validation_results["checks"]["portfolio_weights_sum"] = abs(port_sum - 1.0) < 1e-6
            if abs(port_sum - 1.0) >= 1e-6:
                validation_results["warnings"].append(f"Portfolio weights sum to {port_sum:.6f}, not 1.0")
        
        if benchmark_weights:
            bench_sum = sum(benchmark_weights.values())
            validation_results["checks"]["benchmark_weights_sum"] = abs(bench_sum - 1.0) < 1e-6
            if abs(bench_sum - 1.0) >= 1e-6:
                validation_results["warnings"].append(f"Benchmark weights sum to {bench_sum:.6f}, not 1.0")
        
        # Active weights consistency check
        if portfolio_weights and benchmark_weights and active_weights:
            computed_active = {}
            for asset in portfolio_weights:
                port_w = portfolio_weights.get(asset, 0.0)
                bench_w = benchmark_weights.get(asset, 0.0)
                computed_active[asset] = port_w - bench_w
            
            # Check consistency
            active_consistent = True
            for asset in active_weights:
                if asset in computed_active:
                    diff = abs(active_weights[asset] - computed_active[asset])
                    if diff > 1e-10:
                        active_consistent = False
                        break
            
            validation_results["checks"]["active_weights_consistency"] = active_consistent
            if not active_consistent:
                validation_results["warnings"].append("Active weights inconsistent with portfolio-benchmark")
        
        # Update main validation data
        self._data["validation"].update(validation_results)
        
        return validation_results

    
    def __repr__(self) -> str:
        """String representation of the schema."""
        return (f"RiskResultSchema("
                f"type={self.analysis_type.value}, "
                f"assets={len(self.asset_names)}, "
                f"factors={len(self.factor_names)}, "
                f"timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M')})")