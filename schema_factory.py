"""
Risk Schema Factory
===================

Factory methods for creating unified risk result schemas from various sources.
Provides convenient, standardized ways to create schemas across the Spark system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from .schema import RiskResultSchema, AnalysisType
from .schema_utils import create_schema_from_arrays, SchemaConverter


def _convert_weights_to_dict(weights, fallback_asset_names: List[str] = None) -> Dict[str, float]:
    """Helper function to convert weights array to dictionary with asset names."""
    if isinstance(weights, dict):
        return weights.copy()
    elif hasattr(weights, '__len__') and len(weights) > 0:
        # Convert numpy array or other sequence to dict using generic names
        asset_names = fallback_asset_names or [f'asset_{i}' for i in range(len(weights))]
        return dict(zip(asset_names[:len(weights)], weights))
    else:
        return {}


class RiskSchemaFactory:
    """Factory for creating unified risk result schemas."""
    
    @staticmethod
    def from_decomposer(decomposer) -> RiskResultSchema:
        """
        Create schema from RiskDecomposer instance.
        
        Parameters
        ----------
        decomposer : RiskDecomposer
            Decomposer instance
            
        Returns
        -------
        RiskResultSchema
            Unified schema
        """
        schema = decomposer._create_unified_schema()
        return schema
    
    
    @staticmethod
    def from_arrays(
        total_risk: float,
        factor_contributions: Union[np.ndarray, List[float]],
        asset_contributions: Union[np.ndarray, List[float]],
        factor_exposures: Optional[Union[np.ndarray, List[float]]] = None,
        factor_loadings: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None,
        factor_names: Optional[List[str]] = None,
        analysis_type: str = "portfolio"
    ) -> RiskResultSchema:
        """
        Create schema from raw arrays.
        
        Parameters
        ----------
        total_risk : float
            Total portfolio risk
        factor_contributions : array-like
            Factor risk contributions
        asset_contributions : array-like
            Asset risk contributions
        factor_exposures : array-like, optional
            Portfolio factor exposures
        factor_loadings : array-like, optional
            Asset factor loadings (N×K matrix)
        asset_names : list of str, optional
            Asset names/symbols
        factor_names : list of str, optional
            Factor names
        analysis_type : str, default "portfolio"
            Type of analysis
            
        Returns
        -------
        RiskResultSchema
            Unified schema
        """
        return create_schema_from_arrays(
            total_risk=total_risk,
            factor_contributions=factor_contributions,
            asset_contributions=asset_contributions,
            factor_exposures=factor_exposures,
            factor_loadings=factor_loadings,
            asset_names=asset_names,
            factor_names=factor_names,
            analysis_type=analysis_type
        )
    
    @staticmethod
    def from_portfolio_summary(
        summary: Dict[str, Any],
        analysis_type: str = "hierarchical"
    ) -> RiskResultSchema:
        """
        Create schema from portfolio risk summary.
        
        Parameters
        ----------
        summary : dict
            Portfolio risk summary
        analysis_type : str, default "hierarchical"
            Type of analysis
            
        Returns
        -------
        RiskResultSchema
            Unified schema
        """
        return SchemaConverter.portfolio_summary_to_schema(summary, analysis_type)
    
    @staticmethod
    def from_visitor_results(
        visitor,
        component_id: str,
        analysis_type: str = "hierarchical",
        populate_all_components: bool = False
    ) -> RiskResultSchema:
        """
        Create schema from visitor results.
        
        Parameters
        ----------
        visitor : FactorRiskDecompositionVisitor
            Visitor instance with results
        component_id : str
            Component ID to extract results for
        analysis_type : str, default "hierarchical"
            Type of analysis
        populate_all_components : bool, default False
            If True, populate risk data for all components (not just the specified one)
            
        Returns
        -------
        RiskResultSchema
            Unified schema
        """
        # Extract asset names from visitor's metric store - simplified approach
        asset_names = []
        factor_names = getattr(visitor, 'factor_names', [])
        
        # Get asset names from visitor's metric store (stored during decomposition)
        if hasattr(visitor, 'metric_store') and visitor.metric_store:
            asset_names_metric = visitor.metric_store.get_metric(component_id, 'asset_names')
            if asset_names_metric:
                asset_names = asset_names_metric.value()
                # Convert full paths to component names if needed (e.g., "TOTAL/EQLIKE/EQ/EQDM/EQDMLC" -> "EQDMLC")
                asset_names = [name.split('/')[-1] if '/' in str(name) else str(name) for name in asset_names]
        
        # Create schema with proper asset names
        schema = RiskResultSchema(
            analysis_type=analysis_type,
            asset_names=asset_names,
            factor_names=factor_names
        )
        
        # Set component_ids to asset names if we have them
        if asset_names:
            schema.component_ids = asset_names.copy()
        
        # Try to extract risk decomposition context for the component
        try:
            if hasattr(visitor, 'metric_store'):
                context_metric = visitor.metric_store.get_metric(component_id, 'hierarchical_model_context')
                if context_metric and hasattr(context_metric, 'value'):
                    hierarchical_context = context_metric.value()
                    
                    # Extract data from all three decomposers to populate multi-lens structure
                    
                    # PORTFOLIO LENS
                    if hasattr(hierarchical_context, 'portfolio_decomposer'):
                        portfolio_decomposer = hierarchical_context.portfolio_decomposer
                        
                        # Portfolio lens - comprehensive data
                        schema.set_lens_core_metrics(
                            lens='portfolio',
                            total_risk=portfolio_decomposer.portfolio_volatility,
                            factor_risk_contribution=portfolio_decomposer.factor_risk_contribution,
                            specific_risk_contribution=portfolio_decomposer.specific_risk_contribution
                        )
                        schema.set_lens_asset_contributions('portfolio', portfolio_decomposer.asset_total_contributions)
                        schema.set_lens_factor_contributions('portfolio', portfolio_decomposer.factor_contributions)
                        schema.set_lens_factor_exposures('portfolio', portfolio_decomposer.portfolio_factor_exposure)
                        
                        # Portfolio weights via schema's weights section
                        if hasattr(portfolio_decomposer, '_results') and 'portfolio_weights' in portfolio_decomposer._results:
                            portfolio_weights = portfolio_decomposer._results['portfolio_weights']
                            schema._data['weights']['portfolio_weights'] = _convert_weights_to_dict(portfolio_weights)
                        
                        # Set validation results
                        validation_results = portfolio_decomposer.validate_contributions()
                        schema.set_validation_results(validation_results)
                    
                    # BENCHMARK LENS
                    if hasattr(hierarchical_context, 'benchmark_decomposer'):
                        benchmark_decomposer = hierarchical_context.benchmark_decomposer
                        
                        # Benchmark lens - new structure
                        schema.set_lens_core_metrics(
                            'benchmark',
                            total_risk=benchmark_decomposer.portfolio_volatility,
                            factor_risk_contribution=benchmark_decomposer.factor_risk_contribution,
                            specific_risk_contribution=benchmark_decomposer.specific_risk_contribution
                        )
                        schema.set_lens_factor_exposures('benchmark', benchmark_decomposer.portfolio_factor_exposure)
                        schema.set_lens_asset_contributions('benchmark', benchmark_decomposer.asset_total_contributions)
                        schema.set_lens_factor_contributions('benchmark', benchmark_decomposer.factor_contributions)
                        
                        # Benchmark weights via schema's weights section  
                        if hasattr(benchmark_decomposer, '_results') and 'portfolio_weights' in benchmark_decomposer._results:
                            benchmark_weights = benchmark_decomposer._results['portfolio_weights']
                            schema._data['weights']['benchmark_weights'] = _convert_weights_to_dict(benchmark_weights)
                    
                    # ACTIVE LENS
                    if hasattr(hierarchical_context, 'active_decomposer'):
                        active_decomposer = hierarchical_context.active_decomposer
                        
                        # Active lens - new structure
                        schema.set_lens_core_metrics(
                            'active',
                            total_risk=active_decomposer.portfolio_volatility,
                            factor_risk_contribution=active_decomposer.factor_risk_contribution,
                            specific_risk_contribution=active_decomposer.specific_risk_contribution
                        )
                        schema.set_lens_factor_exposures('active', active_decomposer.portfolio_factor_exposure)
                        schema.set_lens_asset_contributions('active', active_decomposer.asset_total_contributions)
                        schema.set_lens_factor_contributions('active', active_decomposer.factor_contributions)
                        
                        # Active weights calculation via schema's weights section
                        if hasattr(active_decomposer, '_results'):
                            results = active_decomposer._results
                            if 'portfolio_weights' in results and 'benchmark_weights' in results:
                                # Calculate active weights as portfolio - benchmark
                                active_portfolio = results['portfolio_weights']
                                active_benchmark = results['benchmark_weights']
                                if isinstance(active_portfolio, dict) and isinstance(active_benchmark, dict):
                                    active_weights = {asset: active_portfolio.get(asset, 0.0) - active_benchmark.get(asset, 0.0) 
                                                    for asset in set(active_portfolio.keys()) | set(active_benchmark.keys())}
                                    schema._data['weights']['active_weights'] = active_weights
                            elif 'portfolio_weights' in results:
                                # Fallback: use portfolio_weights directly if benchmark not available
                                active_weights = results['portfolio_weights']
                                schema._data['weights']['active_weights'] = _convert_weights_to_dict(active_weights)
                        
                        # Active risk metrics stored in context info for legacy compatibility
                        active_metrics = {
                            'total_active_risk': active_decomposer.portfolio_volatility,
                            'active_factor_risk': active_decomposer.factor_risk_contribution,
                            'active_specific_risk': active_decomposer.specific_risk_contribution
                        }
                        schema.add_context_info('active_risk_metrics', active_metrics)
                        
                        # Add context information
                        schema.add_context_info('component_id', component_id)
                        schema.add_context_info('visitor_type', type(visitor).__name__)
                        schema.add_context_info('extraction_source', 'hierarchical_context')
                        
                        # Extract hierarchy information if available
                        if hasattr(visitor, 'portfolio_graph'):
                            _extract_hierarchy_from_graph(schema, visitor.portfolio_graph, component_id)
                        elif hasattr(visitor, 'graph'):
                            _extract_hierarchy_from_graph(schema, visitor.graph, component_id)
                        else:
                            # Try to find portfolio graph in hierarchical context
                            if hasattr(hierarchical_context, 'portfolio_graph'):
                                _extract_hierarchy_from_graph(schema, hierarchical_context.portfolio_graph, component_id)
                            elif hasattr(hierarchical_context.portfolio_decomposer, 'portfolio_graph'):
                                _extract_hierarchy_from_graph(schema, hierarchical_context.portfolio_decomposer.portfolio_graph, component_id)
                        
                        # Extract time series data if available
                        # Time series data should be provided separately to avoid coupling
                        # The caller can use schema.set_time_series_metadata() and related methods
                        # to populate time series data from their own sources

        except Exception as e:
            raise ValueError(f"Failed to extract hierarchical context for component '{component_id}': {e}")

        # **NEW: Optionally populate all hierarchical components**
        if populate_all_components and analysis_type == "hierarchical":
            try:
                population_summary = schema.populate_hierarchical_risk_data_from_visitor(visitor)
                schema.add_context_info('bulk_population_summary', population_summary)
                
                # Update component_ids list from populated data
                populated_components = schema.get_available_components()
                if populated_components:
                    schema.component_ids = populated_components
                    
            except Exception as population_error:
                schema.add_context_info('bulk_population_error', str(population_error))
        
        return schema
    
    @staticmethod
    def from_visitor_results_with_time_series(
        visitor,
        component_id: str,
        portfolio_returns: Optional[Dict[str, pd.Series]] = None,
        benchmark_returns: Optional[Dict[str, pd.Series]] = None,
        factor_returns: Optional[Dict[str, pd.Series]] = None,
        dates: Optional[List] = None,
        analysis_type: str = "hierarchical"
    ) -> RiskResultSchema:
        """
        Create schema from visitor results with explicit time series data.
        This method avoids coupling by accepting time series data as explicit parameters
        rather than trying to extract it from visitor internals.
        
        Parameters
        ----------
        visitor : FactorRiskDecompositionVisitor
            Visitor instance with results
        component_id : str
            Component ID to extract results for
        portfolio_returns : dict, optional
            Dictionary mapping component IDs to portfolio return series
        benchmark_returns : dict, optional
            Dictionary mapping component IDs to benchmark return series
        factor_returns : dict, optional
            Dictionary mapping factor names to factor return series
        dates : list, optional
            List of dates for the time series
        analysis_type : str, default "hierarchical"
            Type of analysis
            
        Returns
        -------
        RiskResultSchema
            Schema with risk results and time series data
        """
        # First create schema from visitor results
        schema = RiskSchemaFactory.from_visitor_results(visitor, component_id, analysis_type)
        
        # Add time series data if provided
        if dates:
            schema.set_time_series_metadata(dates=dates, frequency="daily")
            
        if portfolio_returns:
            for comp_id, returns in portfolio_returns.items():
                schema.set_component_portfolio_returns(comp_id, returns.values if hasattr(returns, 'values') else returns)
                
        if benchmark_returns:
            for comp_id, returns in benchmark_returns.items():
                schema.set_component_benchmark_returns(comp_id, returns.values if hasattr(returns, 'values') else returns)
                
        if factor_returns:
            for factor_name, returns in factor_returns.items():
                schema.set_factor_returns(factor_name, returns.values if hasattr(returns, 'values') else returns)
        
        return schema
    
    @staticmethod
    def merge_schemas(schemas: List[RiskResultSchema], merged_name: str = "merged") -> RiskResultSchema:
        """
        Merge multiple schemas into a consolidated view.
        
        Parameters
        ----------
        schemas : list of RiskResultSchema
            Schemas to merge
        merged_name : str, default "merged"
            Name identifier for merged result
            
        Returns
        -------
        RiskResultSchema
            Merged schema
        """
        return SchemaConverter.merge_schemas(schemas)
    
    @staticmethod
    def create_empty_schema(
        analysis_type: str = "portfolio",
        asset_names: Optional[List[str]] = None,
        factor_names: Optional[List[str]] = None
    ) -> RiskResultSchema:
        """
        Create an empty schema with basic structure.
        
        Parameters
        ----------
        analysis_type : str, default "portfolio"
            Type of analysis
        asset_names : list of str, optional
            Asset names
        factor_names : list of str, optional
            Factor names
            
        Returns
        -------
        RiskResultSchema
            Empty schema with basic structure
        """
        return RiskResultSchema(
            analysis_type=analysis_type,
            asset_names=asset_names or [],
            factor_names=factor_names or []
        )
    
    @staticmethod
    def validate_and_create(
        data_source: Any,
        source_type: str = "auto"
    ) -> RiskResultSchema:
        """
        Validate source and create appropriate schema.
        
        Parameters
        ----------
        data_source : any
            Source data (decomposer, dict, array, etc.)
        source_type : str, default "auto"
            Type of source ("auto", "decomposer", "dict", "arrays")
            
        Returns
        -------
        RiskResultSchema
            Created schema
        """
        from .decomposer import RiskDecomposer
        
        if source_type == "auto":
            # Auto-detect source type
            if isinstance(data_source, RiskDecomposer):
                return RiskSchemaFactory.from_decomposer(data_source)
            elif isinstance(data_source, dict):
                # Dictionary input - create empty schema with basic data
                schema = RiskSchemaFactory.create_empty_schema()
                if "total_risk" in data_source:
                    # Use lens-specific method for core metrics
                    schema.set_lens_core_metrics(
                        lens='portfolio',  # Default to portfolio lens
                        total_risk=data_source["total_risk"],
                        factor_risk_contribution=data_source.get("factor_risk", 0.0),
                        specific_risk_contribution=data_source.get("specific_risk", 0.0)
                    )
                return schema
            else:
                # Unknown type - return empty schema
                return RiskSchemaFactory.create_empty_schema()
        
        elif source_type == "decomposer":
            return RiskSchemaFactory.from_decomposer(data_source)
        elif source_type == "dict":
            # Dictionary input - create empty schema with basic data
            schema = RiskSchemaFactory.create_empty_schema()
            if "total_risk" in data_source:
                schema.set_lens_core_metrics(
                    lens='portfolio',
                    total_risk=data_source["total_risk"],
                    factor_risk_contribution=data_source.get("factor_risk", 0.0),
                    specific_risk_contribution=data_source.get("specific_risk", 0.0)
                )
            return schema
        else:
            return RiskSchemaFactory.create_empty_schema()
    
    @staticmethod
    def from_visitor_direct_mapping(
        visitor,
        root_component_id: str,
        map_full_hierarchy: bool = True
    ) -> RiskResultSchema:
        """
        Create schema with direct mapping from visitor metric store.
        
        This method provides complete decoupling from visitor internals by only
        accessing the metric store data populated during risk decomposition.
        
        Parameters
        ----------
        visitor : FactorRiskDecompositionVisitor
            Visitor with completed risk analysis and populated metric store
        root_component_id : str
            Root component for the analysis
        map_full_hierarchy : bool, default True
            If True, map all processed components in the hierarchy
            If False, only map the root component
            
        Returns
        -------
        RiskResultSchema
            Schema with hierarchical risk data directly mapped from visitor
        """
        # Extract asset and factor names from visitor
        asset_names = []
        factor_names = getattr(visitor, 'factor_names', [])
        
        # Get asset names from visitor's metric store
        if hasattr(visitor, 'metric_store') and visitor.metric_store:
            asset_names_metric = visitor.metric_store.get_metric(root_component_id, 'asset_names')
            if asset_names_metric:
                asset_names = asset_names_metric.value()
                # Clean asset names (remove path prefixes if present)
                asset_names = [name.split('/')[-1] if '/' in str(name) else str(name) for name in asset_names]
        
        # Create base schema
        schema = RiskResultSchema(
            analysis_type=AnalysisType.HIERARCHICAL,
            asset_names=asset_names,
            factor_names=factor_names
        )
        
        # Extract hierarchical data
        if map_full_hierarchy:
            hierarchical_data = _map_hierarchical_tree(visitor, root_component_id)
        else:
            node_data = _extract_node_risk_data(visitor, root_component_id, asset_names, factor_names)
            hierarchical_data = {root_component_id: node_data} if node_data else {}
        
        # Set hierarchical data in schema
        if hierarchical_data:
            schema.set_hierarchical_risk_data(hierarchical_data)
        
        # Add metadata
        schema.add_context_info('extraction_method', 'direct_mapping')
        schema.add_context_info('visitor_type', type(visitor).__name__)
        schema.add_context_info('root_component', root_component_id)
        schema.add_context_info('components_mapped', len(hierarchical_data))
        schema.add_context_info('mapping_mode', 'full_hierarchy' if map_full_hierarchy else 'single_component')
        
        return schema


def _extract_decomposer_data(
    decomposer,
    asset_names: List[str], 
    factor_names: List[str]
) -> Dict[str, Any]:
    """
    Extract comprehensive data from a single risk decomposer using direct property access.
    
    This function extracts ALL available properties from the decomposer instance,
    providing complete data coverage without going through summary dictionaries.
    
    Parameters
    ----------
    decomposer : RiskDecomposer
        Risk decomposer instance
    asset_names : list of str
        Asset names for mapping contributions
    factor_names : list of str
        Factor names for mapping contributions
        
    Returns
    -------
    dict
        Comprehensive lens data structure with all decomposer properties
    """
    lens_data = {}
    
    try:
        # Core risk metrics - using direct property access

        lens_data['total_risk'] = decomposer.portfolio_volatility
        lens_data['factor_risk_contribution'] = decomposer.factor_risk_contribution
        lens_data['specific_risk_contribution'] = decomposer.specific_risk_contribution
        lens_data['factor_risk_percentage'] = (decomposer.factor_risk_contribution / decomposer.portfolio_volatility * 100) if decomposer.portfolio_volatility > 0 else 0.0
        lens_data['specific_risk_percentage'] = (decomposer.specific_risk_contribution / decomposer.portfolio_volatility * 100) if decomposer.portfolio_volatility > 0 else 0.0

        
        # Extract core array properties and convert to named dictionaries
        _extract_array_properties(lens_data, decomposer, asset_names, factor_names)
        
        # Extract optional active risk properties
        _extract_active_risk_properties(lens_data, decomposer, asset_names, factor_names)
        
        # Extract matrix data and additional properties
        _extract_matrix_properties(lens_data, decomposer, asset_names, factor_names)
        
        # Extract percentage properties
        _extract_percentage_properties(lens_data, decomposer, asset_names, factor_names)
        
    except Exception as e:
        # Return partial data if extraction fails
        lens_data['extraction_error'] = str(e)
    
    return lens_data


def _extract_array_properties(
    lens_data: Dict[str, Any],
    decomposer,
    asset_names: List[str], 
    factor_names: List[str]
) -> None:
    """Extract core array properties and convert to named dictionaries."""
    
    # Factor contributions
    try:
        factor_contributions = decomposer.factor_contributions
        if factor_contributions is not None and len(factor_contributions) > 0:
            if len(factor_names) == len(factor_contributions):
                lens_data['factor_contributions'] = dict(zip(factor_names, factor_contributions))
            else:
                # Fallback: use indices if names don't match
                lens_data['factor_contributions'] = {f'factor_{i}': val for i, val in enumerate(factor_contributions)}
    except Exception:
        pass
    
    # Asset total contributions
    try:
        asset_contributions = decomposer.asset_total_contributions
        if asset_contributions is not None and len(asset_contributions) > 0:
            if len(asset_names) == len(asset_contributions):
                lens_data['asset_contributions'] = dict(zip(asset_names, asset_contributions))
            else:
                # Fallback: use indices if names don't match
                lens_data['asset_contributions'] = {f'asset_{i}': val for i, val in enumerate(asset_contributions)}
    except Exception:
        pass
    
    # Portfolio factor exposures
    try:
        factor_exposures = decomposer.portfolio_factor_exposure
        if factor_exposures is not None and len(factor_exposures) > 0:
            if len(factor_names) == len(factor_exposures):
                lens_data['factor_exposures'] = dict(zip(factor_names, factor_exposures))
            else:
                lens_data['factor_exposures'] = {f'factor_{i}': val for i, val in enumerate(factor_exposures)}
    except Exception:
        pass
    
    # Portfolio weights
    try:
        portfolio_weights = decomposer.portfolio_weights
        if portfolio_weights is not None and len(portfolio_weights) > 0:
            if len(asset_names) == len(portfolio_weights):
                lens_data['portfolio_weights'] = dict(zip(asset_names, portfolio_weights))
            else:
                lens_data['portfolio_weights'] = {f'asset_{i}': val for i, val in enumerate(portfolio_weights)}
    except Exception:
        pass
    
    # Marginal contributions
    try:
        marginal_factor = decomposer.marginal_factor_contributions
        if marginal_factor is not None and len(marginal_factor) > 0:
            if len(factor_names) == len(marginal_factor):
                lens_data['marginal_factor_contributions'] = dict(zip(factor_names, marginal_factor))
    except Exception:
        pass
    
    try:
        marginal_asset = decomposer.marginal_asset_contributions  
        if marginal_asset is not None and len(marginal_asset) > 0:
            if len(asset_names) == len(marginal_asset):
                lens_data['marginal_asset_contributions'] = dict(zip(asset_names, marginal_asset))
    except Exception:
        pass


def _extract_active_risk_properties(
    lens_data: Dict[str, Any],
    decomposer,
    asset_names: List[str], 
    factor_names: List[str]
) -> None:
    """Extract optional active risk properties when available."""
    
    # Active risk scalar properties
    active_scalars = [
        'total_active_risk', 'allocation_factor_risk', 'allocation_specific_risk',
        'selection_factor_risk', 'selection_specific_risk', 'total_allocation_risk',
        'total_selection_risk'
    ]
    
    for prop_name in active_scalars:
        try:
            value = getattr(decomposer, prop_name, None)
            if value is not None:
                lens_data[prop_name] = value
        except Exception:
            pass
    
    # Active risk array properties
    active_arrays = [
        ('benchmark_weights', asset_names),
        ('active_weights', asset_names),
        ('benchmark_factor_exposure', factor_names),
        ('active_factor_exposure', factor_names),
        ('asset_allocation_factor_contributions', asset_names),
        ('asset_allocation_specific_contributions', asset_names),
        ('asset_selection_factor_contributions', asset_names),
        ('asset_selection_specific_contributions', asset_names),
        ('factor_allocation_contributions', factor_names),
        ('factor_selection_contributions', factor_names),
        ('asset_factor_contributions', asset_names),
        ('asset_specific_contributions', asset_names)
    ]
    
    for prop_name, names in active_arrays:
        try:
            value = getattr(decomposer, prop_name, None)
            if value is not None and len(value) > 0:
                if len(names) == len(value):
                    lens_data[prop_name] = dict(zip(names, value))
                else:
                    lens_data[prop_name] = {f'{prop_name}_{i}': val for i, val in enumerate(value)}
        except Exception:
            pass


def _extract_matrix_properties(
    lens_data: Dict[str, Any],
    decomposer,
    asset_names: List[str], 
    factor_names: List[str]
) -> None:
    """Extract matrix data and special properties."""
    
    # Asset by factor contributions matrix
    try:
        matrix = decomposer.asset_by_factor_contributions
        if matrix is not None:
            lens_data['asset_by_factor_contributions'] = _convert_matrix_to_dict(matrix, asset_names, factor_names)
    except Exception:
        pass
    
    # Weighted betas from results dictionary
    try:
        if hasattr(decomposer, 'results'):
            results = decomposer.results
            weighted_betas = results.get('weighted_betas')
            if weighted_betas is not None:
                if isinstance(weighted_betas, dict):
                    lens_data['weighted_betas'] = weighted_betas
                else:
                    lens_data['weighted_betas'] = _convert_matrix_to_dict(weighted_betas, asset_names, factor_names)
    except Exception:
        pass
    
    # Risk decomposition dictionary
    try:
        risk_decomp = decomposer.risk_decomposition
        if risk_decomp is not None:
            lens_data['risk_decomposition'] = risk_decomp
    except Exception:
        pass


def _extract_percentage_properties(
    lens_data: Dict[str, Any],
    decomposer,
    asset_names: List[str], 
    factor_names: List[str]
) -> None:
    """Extract percentage-based properties."""
    
    try:
        percent_total = decomposer.percent_total_contributions
        if percent_total is not None and len(percent_total) > 0:
            if len(asset_names) == len(percent_total):
                lens_data['percent_total_contributions'] = dict(zip(asset_names, percent_total))
    except Exception:
        pass
    
    try:
        percent_factor = decomposer.percent_factor_contributions
        if percent_factor is not None and len(percent_factor) > 0:
            if len(factor_names) == len(percent_factor):
                lens_data['percent_factor_contributions'] = dict(zip(factor_names, percent_factor))
    except Exception:
        pass


def _extract_lens_data_from_context(
    context,
    lens_name: str,
    asset_names: List[str], 
    factor_names: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Extract lens-specific data from hierarchical model context.
    
    Parameters
    ----------
    context : HierarchicalModelContext
        Hierarchical model context with decomposers
    lens_name : str
        Name of lens ('portfolio', 'benchmark', 'active')
    asset_names : list of str
        Asset names for mapping contributions
    factor_names : list of str
        Factor names for mapping contributions
        
    Returns
    -------
    dict or None
        Lens data structure or None if extraction fails
    """
    decomposer_attr = f'{lens_name}_decomposer'
    
    if not hasattr(context, decomposer_attr):
        return None
        
    try:
        decomposer = getattr(context, decomposer_attr)
        return _extract_decomposer_data(decomposer, asset_names, factor_names)
    except Exception as e:
        # Return None if lens extraction fails - skip this lens
        return None


def _extract_node_risk_data(
    visitor, 
    component_id: str, 
    asset_names: List[str], 
    factor_names: List[str]
) -> Dict[str, Any]:
    """
    Extract all risk data for a single component from visitor metric store.
    
    This function provides the core mapping logic to extract decomposer results
    from the visitor's metric store without coupling to visitor internals.
    
    Parameters
    ----------
    visitor : FactorRiskDecompositionVisitor
        Visitor with populated metric store
    component_id : str
        Component to extract data for
    asset_names : list of str
        Asset names for mapping contributions
    factor_names : list of str
        Factor names for mapping contributions
        
    Returns
    -------
    dict
        Nested dictionary with risk data for all available lenses
    """
    node_data = {}
    
    try:
        # Get hierarchical model context from metric store
        context_metric = visitor.metric_store.get_metric(component_id, 'hierarchical_model_context')
        if not context_metric:
            # This might be a leaf component - check for individual risk models
            portfolio_model = visitor.metric_store.get_metric(component_id, 'portfolio_risk_model')
            benchmark_model = visitor.metric_store.get_metric(component_id, 'benchmark_risk_model') 
            active_model = visitor.metric_store.get_metric(component_id, 'active_risk_model')
            
            if portfolio_model or benchmark_model or active_model:
                # Extract data from individual models (leaf component)
                return _extract_from_individual_models(visitor, component_id, asset_names, factor_names, 
                                                     portfolio_model, benchmark_model, active_model)
            else:
                return node_data
            
        context = context_metric.value()
        
        # Extract data for each lens using helper functions
        for lens_name in ['portfolio', 'benchmark', 'active']:
            lens_data = _extract_lens_data_from_context(context, lens_name, asset_names, factor_names)
            if lens_data:
                node_data[lens_name] = lens_data
    
    except Exception as e:
        # Return empty data if extraction fails - don't break the entire process
        node_data = {}
    
    return node_data


def _map_hierarchical_tree(visitor, root_component_id: str) -> Dict[str, Dict]:
    """
    Map entire component tree from visitor metric store data.
    
    Parameters
    ----------
    visitor : FactorRiskDecompositionVisitor
        Visitor with populated metric store
    root_component_id : str
        Root component for the tree
        
    Returns
    -------
    dict
        Dictionary mapping component_id -> risk_data for all processed components
    """
    hierarchical_data = {}
    
    # Get asset and factor names for mapping
    asset_names = []
    factor_names = getattr(visitor, 'factor_names', [])
    
    if hasattr(visitor, 'metric_store') and visitor.metric_store:
        asset_names_metric = visitor.metric_store.get_metric(root_component_id, 'asset_names')
        if asset_names_metric:
            asset_names = asset_names_metric.value()
            asset_names = [name.split('/')[-1] if '/' in str(name) else str(name) for name in asset_names]
    
    # Get all processed components from visitor
    processed_components = []
    if hasattr(visitor, 'get_processed_components'):
        processed_components = list(visitor.get_processed_components())
    elif hasattr(visitor, '_processed_components'):
        processed_components = list(visitor._processed_components)
    
    # If no processed components found, try to find all components with hierarchical_model_context
    if not processed_components and hasattr(visitor, 'metric_store') and visitor.metric_store:
        # We need a different approach since metric_store doesn't expose internal structure
        # Try to get components from the portfolio graph if available
        if hasattr(visitor, 'portfolio_graph') and visitor.portfolio_graph:
            for comp_id in visitor.portfolio_graph.components.keys():
                if visitor.metric_store.get_metric(comp_id, 'hierarchical_model_context'):
                    processed_components.append(comp_id)
    
    # Extract data for all processed components
    for component_id in processed_components:
        node_data = _extract_node_risk_data(visitor, component_id, asset_names, factor_names)
        if node_data:
            hierarchical_data[component_id] = node_data
    
    return hierarchical_data


def _convert_matrix_to_dict(matrix, row_names: List[str], col_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Convert a numpy matrix to nested dictionary format for UI consumption.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to convert
    row_names : list of str
        Names for matrix rows
    col_names : list of str
        Names for matrix columns
        
    Returns
    -------
    dict
        Nested dictionary: {row_name: {col_name: value}}
    """
    if matrix is None or matrix.size == 0:
        return {}
    
    try:
        import numpy as np
        
        if matrix.ndim == 2:
            n_rows, n_cols = matrix.shape
            
            # Use provided names or generate defaults
            if len(row_names) != n_rows:
                row_names = [f"row_{i}" for i in range(n_rows)]
            if len(col_names) != n_cols:
                col_names = [f"col_{i}" for i in range(n_cols)]
            
            return {
                row_name: {
                    col_name: float(matrix[i, j])
                    for j, col_name in enumerate(col_names)
                }
                for i, row_name in enumerate(row_names)
            }
        elif matrix.ndim == 1:
            # Vector case - convert to single row
            if len(col_names) == len(matrix):
                return {
                    "values": {
                        col_name: float(matrix[i])
                        for i, col_name in enumerate(col_names)
                    }
                }
        
        return {}
        
    except Exception:
        return {}


# Convenience functions for quick schema creation
def create_risk_schema_from_decomposer(decomposer) -> RiskResultSchema:
    """Convenience function to create schema from decomposer."""
    return RiskSchemaFactory.from_decomposer(decomposer)


def create_risk_schema_from_arrays(
    total_risk: float,
    factor_contributions: Union[np.ndarray, List[float]],
    asset_contributions: Union[np.ndarray, List[float]],
    **kwargs
) -> RiskResultSchema:
    """Convenience function to create schema from arrays."""
    return RiskSchemaFactory.from_arrays(total_risk, factor_contributions, asset_contributions, **kwargs)


def create_empty_risk_schema(analysis_type: str = "portfolio") -> RiskResultSchema:
    """Convenience function to create empty schema."""
    return RiskSchemaFactory.create_empty_schema(analysis_type)


def _extract_hierarchy_from_graph(schema: RiskResultSchema, portfolio_graph, root_component_id: str) -> None:
    """Extract hierarchy information from a portfolio graph."""
    try:
        # Check if the graph has the necessary methods
        if not hasattr(portfolio_graph, 'components') or not hasattr(portfolio_graph, 'adjacency_list'):
            return
        
        components = portfolio_graph.components
        adjacency_list = portfolio_graph.adjacency_list
        
        # Build component relationships
        component_relationships = {}
        component_metadata = {}
        
        for component_id, component in components.items():
            # Initialize relationships
            children = adjacency_list.get(component_id, [])
            parent = None
            
            # Find parent by looking through adjacency list
            for potential_parent, child_list in adjacency_list.items():
                if component_id in child_list:
                    parent = potential_parent
                    break
            
            component_relationships[component_id] = {
                "parent": parent,
                "children": children
            }
            
            # Extract component metadata
            metadata = {
                "component_id": component_id,
                "type": "leaf" if not children else "node"
            }
            
            # Try to extract additional metadata from component
            if hasattr(component, 'component_type'):
                metadata["component_type"] = component.component_type
            if hasattr(component, 'is_overlay'):
                metadata["is_overlay"] = component.is_overlay
            if hasattr(component, 'data') and hasattr(component.data, 'keys'):
                metadata["data_keys"] = list(component.data.keys())
            
            # Calculate level and path
            if component_id == root_component_id:
                metadata["level"] = 0
                metadata["path"] = component_id
            else:
                # Calculate path and level
                path_parts = component_id.split('/')
                metadata["level"] = len(path_parts) - 1
                metadata["path"] = component_id
            
            component_metadata[component_id] = metadata
        
        # Set hierarchy structure in schema
        schema.set_hierarchy_structure(
            root_component=root_component_id,
            component_relationships=component_relationships,
            component_metadata=component_metadata,
            adjacency_list=adjacency_list
        )
        
        # Add hierarchy summary to context
        hierarchy_summary = schema.get_hierarchy_summary()
        schema.add_context_info('hierarchy_summary', hierarchy_summary)
        
    except Exception as e:
        # If hierarchy extraction fails, continue without it
        schema.add_context_info('hierarchy_extraction_error', str(e))


def _extract_time_series_from_visitor(schema: RiskResultSchema, visitor, root_component_id: str, portfolio_graph=None) -> None:
    """Extract time series data from visitor and populate schema."""
    try:
        import pandas as pd
        import numpy as np
        
        # Check if visitor has portfolio graph with time series data
        portfolio_graph = getattr(visitor, 'portfolio_graph', None) or getattr(visitor, 'graph', None)
        if not portfolio_graph:
            schema.add_context_info('time_series_extraction_error', 'No portfolio graph found in visitor')
            return
        
        # Try to extract available metric names to debug what's actually stored
        debug_info = {}
        if hasattr(portfolio_graph, 'metric_store') and hasattr(portfolio_graph, 'components'):
            sample_component = next(iter(portfolio_graph.components.keys()), None)
            if sample_component:
                # Get all metric names for the first component to understand what's available
                all_metrics = portfolio_graph.metric_store.get_all_metrics(sample_component)
                if all_metrics:
                    debug_info['available_metrics'] = list(all_metrics.keys())
                    
        schema.add_context_info('metric_store_debug', debug_info)
        
        # Extract dates from the visitor if available
        dates = []
        if hasattr(visitor, 'dates'):
            dates = visitor.dates
        elif hasattr(visitor, 'context') and hasattr(visitor.context, 'dates'):
            dates = visitor.context.dates
        elif hasattr(portfolio_graph, 'metric_store'):
            # Try to extract dates from metric store - look for time series data
            for comp_id in portfolio_graph.components:
                # Try common metric names for returns that might contain time series
                for metric_name in ['portfolio_return', 'port_ret', 'returns', 'portfolio_returns']:
                    returns_metric = portfolio_graph.metric_store.get_metric(comp_id, metric_name)
                    if returns_metric and hasattr(returns_metric, 'value'):
                        returns_data = returns_metric.value()
                        if isinstance(returns_data, pd.Series) and hasattr(returns_data, 'index'):
                            dates = returns_data.index.tolist()
                            break
                if dates:
                    break
        
        # Set time series metadata if we have dates
        if dates:
            schema.set_time_series_metadata(
                dates=dates,
                frequency="daily",  # Default assumption
                currency="USD",     # Default assumption
                return_type="simple"
            )
        
        # Extract time series for all components in the portfolio graph
        if hasattr(portfolio_graph, 'components') and hasattr(portfolio_graph, 'metric_store'):
            components = portfolio_graph.components
            metric_store = portfolio_graph.metric_store
            
            # Define possible metric names for different return types
            portfolio_metric_names = ['portfolio_return', 'port_ret', 'returns', 'portfolio_returns']
            benchmark_metric_names = ['benchmark_return', 'bench_ret', 'benchmark_returns']
            active_metric_names = ['active_return', 'excess_return', 'active_returns']
            
            # Extract returns for all components
            for component_id, component in components.items():
                # Portfolio returns - try different possible metric names
                for metric_name in portfolio_metric_names:
                    port_returns_metric = metric_store.get_metric(component_id, metric_name)
                    if port_returns_metric and hasattr(port_returns_metric, 'value'):
                        returns_data = port_returns_metric.value()
                        if isinstance(returns_data, pd.Series):
                            schema.set_component_portfolio_returns(component_id, returns_data.values)
                            break
                        elif isinstance(returns_data, (list, tuple, np.ndarray)):
                            schema.set_component_portfolio_returns(component_id, returns_data)
                            break
                
                # Benchmark returns - try different possible metric names
                for metric_name in benchmark_metric_names:
                    bench_returns_metric = metric_store.get_metric(component_id, metric_name)
                    if bench_returns_metric and hasattr(bench_returns_metric, 'value'):
                        returns_data = bench_returns_metric.value()
                        if isinstance(returns_data, pd.Series):
                            schema.set_component_benchmark_returns(component_id, returns_data.values)
                            break
                        elif isinstance(returns_data, (list, tuple, np.ndarray)):
                            schema.set_component_benchmark_returns(component_id, returns_data)
                            break
                
                # Active returns - try different possible metric names
                for metric_name in active_metric_names:
                    active_returns_metric = metric_store.get_metric(component_id, metric_name)
                    if active_returns_metric and hasattr(active_returns_metric, 'value'):
                        returns_data = active_returns_metric.value()
                        if isinstance(returns_data, pd.Series):
                            schema.set_component_active_returns(component_id, returns_data.values)
                            break
                        elif isinstance(returns_data, (list, tuple, np.ndarray)):
                            schema.set_component_active_returns(component_id, returns_data)
                            break
        
        # Extract factor returns
        if hasattr(visitor, 'factor_returns'):
            factor_returns = visitor.factor_returns
            if isinstance(factor_returns, pd.DataFrame):
                for factor_name in factor_returns.columns:
                    factor_series = factor_returns[factor_name]
                    schema.set_factor_returns(factor_name, factor_series.values)
            elif isinstance(factor_returns, dict):
                for factor_name, returns_data in factor_returns.items():
                    if isinstance(returns_data, pd.Series):
                        schema.set_factor_returns(factor_name, returns_data.values)
                    elif isinstance(returns_data, (list, tuple)):
                        schema.set_factor_returns(factor_name, returns_data)
        
        # Extract factor returns from context if not found in visitor
        elif hasattr(visitor, 'context') and hasattr(visitor.context, 'factor_returns'):
            factor_returns = visitor.context.factor_returns
            if isinstance(factor_returns, pd.DataFrame):
                for factor_name in factor_returns.columns:
                    factor_series = factor_returns[factor_name]
                    schema.set_factor_returns(factor_name, factor_series.values)
            elif isinstance(factor_returns, dict):
                for factor_name, returns_data in factor_returns.items():
                    if isinstance(returns_data, pd.Series):
                        schema.set_factor_returns(factor_name, returns_data.values)
                    elif isinstance(returns_data, (list, tuple)):
                        schema.set_factor_returns(factor_name, returns_data)
        
        # Calculate correlations between time series
        schema.calculate_correlations()
        
        # Add time series extraction context
        schema.add_context_info('time_series_extraction_source', 'visitor_and_portfolio_graph')
        
    except Exception as e:
        # If time series extraction fails, continue without it
        import traceback
        schema.add_context_info('time_series_extraction_error', str(e))
        schema.add_context_info('time_series_extraction_traceback', traceback.format_exc())


def _extract_from_individual_models(
    visitor, 
    component_id: str, 
    asset_names: List[str], 
    factor_names: List[str],
    portfolio_model, 
    benchmark_model, 
    active_model
) -> Dict[str, Any]:
    """
    Extract risk data from individual models stored in metric store (for leaf components).
    
    Parameters
    ----------
    visitor : FactorRiskDecompositionVisitor
        Visitor with populated metric store
    component_id : str
        Component to extract data for
    asset_names : list of str
        Asset names for mapping contributions
    factor_names : list of str
        Factor names for mapping contributions
    portfolio_model, benchmark_model, active_model
        Individual model metrics from metric store
        
    Returns
    -------
    dict
        Nested dictionary with risk data for available lenses
    """
    node_data = {}
    
    try:
        # Import RiskDecomposer and context creators
        from .decomposer import RiskDecomposer
        from .context import create_single_model_context, create_active_risk_context
        
        # Extract portfolio lens data if portfolio model exists
        if portfolio_model:
            try:
                model = portfolio_model.value()
                # For leaf components, we need to get weights - they should be 1.0 for the single asset
                weights = np.array([1.0])  # Leaf components represent single assets with weight 1.0
                
                # Create single model context and decomposer
                context = create_single_model_context(model, weights, annualize=True)
                decomposer = RiskDecomposer(context)
                
                # Extract risk data similar to hierarchical approach
                node_data['portfolio'] = {
                    'decomposer_results': {
                        'total_risk': decomposer.portfolio_volatility or 0.0,
                        'factor_risk_contribution': decomposer.factor_risk_contribution or 0.0,
                        'specific_risk_contribution': decomposer.specific_risk_contribution or 0.0,
                        'factor_risk_percentage': (decomposer.factor_risk_contribution / decomposer.portfolio_volatility * 100) if decomposer.portfolio_volatility and decomposer.portfolio_volatility > 0 else 0.0,
                        'specific_risk_percentage': (decomposer.specific_risk_contribution / decomposer.portfolio_volatility * 100) if decomposer.portfolio_volatility and decomposer.portfolio_volatility > 0 else 0.0,
                    }
                }
                
                # Add factor contributions if available
                if hasattr(decomposer, 'factor_contributions') and decomposer.factor_contributions is not None:
                    factor_contributions = decomposer.factor_contributions
                    if hasattr(factor_contributions, '__len__') and len(factor_names) == len(factor_contributions):
                        node_data['portfolio']['factor_contributions'] = dict(zip(factor_names, factor_contributions))
                
                # Add factor exposures if available
                if hasattr(decomposer, 'portfolio_factor_exposure') and decomposer.portfolio_factor_exposure is not None:
                    factor_exposures = decomposer.portfolio_factor_exposure
                    if hasattr(factor_exposures, '__len__') and len(factor_names) == len(factor_exposures):
                        node_data['portfolio']['factor_exposures'] = dict(zip(factor_names, factor_exposures))
                        
            except Exception as e:
                # Log error but continue
                node_data['portfolio'] = {'error': f'Failed to extract portfolio data: {str(e)}'}
        
        # Extract benchmark lens data if benchmark model exists  
        if benchmark_model:
            try:
                model = benchmark_model.value()
                weights = np.array([1.0])
                
                context = create_single_model_context(model, weights, annualize=True)
                decomposer = RiskDecomposer(context)
                
                node_data['benchmark'] = {
                    'decomposer_results': {
                        'total_risk': decomposer.portfolio_volatility or 0.0,
                        'factor_risk_contribution': decomposer.factor_risk_contribution or 0.0,
                        'specific_risk_contribution': decomposer.specific_risk_contribution or 0.0,
                    }
                }
                
                if hasattr(decomposer, 'factor_contributions') and decomposer.factor_contributions is not None:
                    factor_contributions = decomposer.factor_contributions
                    if hasattr(factor_contributions, '__len__') and len(factor_names) == len(factor_contributions):
                        node_data['benchmark']['factor_contributions'] = dict(zip(factor_names, factor_contributions))
                        
            except Exception as e:
                node_data['benchmark'] = {'error': f'Failed to extract benchmark data: {str(e)}'}
        
        # Extract active lens data if both portfolio and benchmark models exist
        if portfolio_model and benchmark_model:
            try:
                port_model = portfolio_model.value()
                bench_model = benchmark_model.value()
                port_weights = np.array([1.0])
                bench_weights = np.array([1.0])
                
                context = create_active_risk_context(
                    port_model, bench_model, port_weights, bench_weights, 
                    active_model.value() if active_model else None, annualize=True
                )
                decomposer = RiskDecomposer(context)
                
                node_data['active'] = {
                    'decomposer_results': {
                        'total_risk': decomposer.portfolio_volatility or 0.0,
                        'factor_risk_contribution': decomposer.factor_risk_contribution or 0.0,
                        'specific_risk_contribution': decomposer.specific_risk_contribution or 0.0,
                    }
                }
                
                if hasattr(decomposer, 'factor_contributions') and decomposer.factor_contributions is not None:
                    factor_contributions = decomposer.factor_contributions
                    if hasattr(factor_contributions, '__len__') and len(factor_names) == len(factor_contributions):
                        node_data['active']['factor_contributions'] = dict(zip(factor_names, factor_contributions))
                        
            except Exception as e:
                node_data['active'] = {'error': f'Failed to extract active data: {str(e)}'}
                
    except Exception as e:
        # Return partial data with error info
        node_data['extraction_error'] = str(e)
    
    return node_data