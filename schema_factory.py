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
    def from_decomposer_dict(decomposer_dict: Dict[str, Any]) -> RiskResultSchema:
        """
        Create schema from RiskDecomposer.to_dict() result.
        
        Parameters
        ----------
        decomposer_dict : dict
            Result from RiskDecomposer.to_dict()
            
        Returns
        -------
        RiskResultSchema
            Unified schema
        """
        return SchemaConverter.decomposer_to_schema(decomposer_dict)
    
    @staticmethod
    def from_strategy_result(strategy_result: Dict[str, Any]) -> RiskResultSchema:
        """
        Create schema from strategy analysis result.
        
        Parameters
        ----------
        strategy_result : dict
            Result from Strategy.analyze()
            
        Returns
        -------
        RiskResultSchema
            Unified schema
        """
        return SchemaConverter.strategy_to_schema(strategy_result)
    
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
        analysis_type: str = "hierarchical"
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
            
        Returns
        -------
        RiskResultSchema
            Unified schema
        """
        # Extract asset names from visitor or context
        asset_names = []
        factor_names = getattr(visitor, 'factor_names', [])
        
        # Try to get asset names from visitor first
        if hasattr(visitor, 'descendant_leaves') and visitor.descendant_leaves:
            # Convert full paths to component names (e.g., "TOTAL/EQLIKE/EQ/EQDM/EQDMLC" -> "EQDMLC")
            asset_names = [name.split('/')[-1] for name in visitor.descendant_leaves]
        
        # Create schema with proper asset names
        schema = RiskResultSchema(
            analysis_type=analysis_type,
            asset_names=asset_names,
            factor_names=factor_names
        )
        
        # Try to extract risk decomposition context for the component
        try:
            if hasattr(visitor, 'metric_store'):
                context_metric = visitor.metric_store.get_metric(component_id, 'hierarchical_model_context')
                if context_metric and hasattr(context_metric, 'value'):
                    hierarchical_context = context_metric.value()
                    
                    # Try to get better asset names from the context
                    if hasattr(hierarchical_context, 'portfolio_decomposer'):
                        decomposer = hierarchical_context.portfolio_decomposer
                        
                        # Extract asset names from various sources
                        context_asset_names = []
                        
                        # Try from context
                        if hasattr(decomposer, 'context') and hasattr(decomposer.context, 'get_asset_names'):
                            context_asset_names = decomposer.context.get_asset_names()
                        
                        # Try from results structure
                        if not context_asset_names and hasattr(decomposer, '_results'):
                            if 'asset_names' in decomposer._results:
                                context_asset_names = decomposer._results['asset_names']
                            elif 'portfolio_weights' in decomposer._results:
                                if isinstance(decomposer._results['portfolio_weights'], dict):
                                    context_asset_names = list(decomposer._results['portfolio_weights'].keys())
                        
                        # Try from asset contributions
                        if not context_asset_names and hasattr(decomposer, 'asset_total_contributions'):
                            if isinstance(decomposer.asset_total_contributions, dict):
                                context_asset_names = list(decomposer.asset_total_contributions.keys())
                        
                        if context_asset_names:
                            # Convert full paths to component names if needed and update schema
                            asset_names = [name.split('/')[-1] if '/' in str(name) else str(name) for name in context_asset_names]
                            schema.asset_names = asset_names
                            # Also set component_ids to full paths
                            schema.component_ids = [str(name) for name in context_asset_names]
                    
                    # Extract data from all three decomposers to populate multi-lens structure
                    
                    # PORTFOLIO LENS
                    if hasattr(hierarchical_context, 'portfolio_decomposer'):
                        portfolio_decomposer = hierarchical_context.portfolio_decomposer
                        
                        # Portfolio lens - new structure
                        schema.set_lens_core_metrics(
                            'portfolio',
                            total_risk=portfolio_decomposer.portfolio_volatility,
                            factor_risk_contribution=portfolio_decomposer.factor_risk_contribution,
                            specific_risk_contribution=portfolio_decomposer.specific_risk_contribution
                        )
                        schema.set_lens_factor_exposures('portfolio', portfolio_decomposer.portfolio_factor_exposure)
                        schema.set_lens_asset_contributions('portfolio', portfolio_decomposer.asset_total_contributions)
                        schema.set_lens_factor_contributions('portfolio', portfolio_decomposer.factor_contributions)
                        
                        # Legacy structure (backward compatibility)
                        schema.set_core_metrics(
                            total_risk=portfolio_decomposer.portfolio_volatility,
                            factor_risk_contribution=portfolio_decomposer.factor_risk_contribution,
                            specific_risk_contribution=portfolio_decomposer.specific_risk_contribution
                        )
                        schema.set_asset_contributions(portfolio_decomposer.asset_total_contributions)
                        schema.set_factor_contributions(portfolio_decomposer.factor_contributions)
                        schema.set_factor_exposures(portfolio_decomposer.portfolio_factor_exposure)
                        
                        # Portfolio weights
                        if 'portfolio_weights' in portfolio_decomposer._results:
                            schema.set_portfolio_weights(portfolio_decomposer._results['portfolio_weights'])
                        
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
                        
                        # Benchmark weights (stored in portfolio_weights key)
                        if 'portfolio_weights' in benchmark_decomposer._results:
                            schema.set_benchmark_weights(benchmark_decomposer._results['portfolio_weights'])
                    
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
                        
                        # Active weights calculation
                        if ('portfolio_weights' in active_decomposer._results and 
                            'benchmark_weights' in active_decomposer._results):
                            active_portfolio = active_decomposer._results['portfolio_weights']
                            active_benchmark = active_decomposer._results['benchmark_weights']
                            active_weights = active_portfolio - active_benchmark
                            schema.set_active_weights(active_weights, auto_calculate=False)
                        elif 'portfolio_weights' in active_decomposer._results:
                            # Fallback: use portfolio_weights directly if benchmark not available
                            schema.set_active_weights(active_decomposer._results['portfolio_weights'], auto_calculate=False)
                        
                        # Legacy active risk metrics for backward compatibility
                        active_metrics = {
                            'total_active_risk': active_decomposer.portfolio_volatility,
                            'active_factor_risk': active_decomposer.factor_risk_contribution,
                            'active_specific_risk': active_decomposer.specific_risk_contribution
                        }
                        schema.set_active_risk_metrics(active_metrics)
                        
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
        
        except Exception:
            # Fallback to empty schema if extraction fails
            pass
        
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
                # Try decomposer format first, then strategy format
                if "core_metrics" in data_source and "named_contributions" in data_source:
                    return RiskSchemaFactory.from_decomposer_dict(data_source)
                elif "portfolio_volatility" in data_source and "factor_contributions" in data_source:
                    return RiskSchemaFactory.from_strategy_result(data_source)
                else:
                    # Generic dictionary - create empty schema and populate
                    schema = RiskSchemaFactory.create_empty_schema()
                    if "total_risk" in data_source:
                        schema.set_core_metrics(
                            data_source["total_risk"],
                            data_source.get("factor_risk", 0.0),
                            data_source.get("specific_risk", 0.0)
                        )
                    return schema
            else:
                # Unknown type - return empty schema
                return RiskSchemaFactory.create_empty_schema()
        
        elif source_type == "decomposer":
            return RiskSchemaFactory.from_decomposer(data_source)
        elif source_type == "dict":
            return RiskSchemaFactory.from_decomposer_dict(data_source)
        else:
            return RiskSchemaFactory.create_empty_schema()


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