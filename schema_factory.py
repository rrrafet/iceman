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
                        if hasattr(decomposer, 'context') and hasattr(decomposer.context, 'get_asset_names'):
                            context_asset_names = decomposer.context.get_asset_names()
                            if context_asset_names:
                                # Convert full paths to component names if needed
                                asset_names = [name.split('/')[-1] for name in context_asset_names]
                                # Update schema with better asset names
                                schema.asset_names = asset_names
                    
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
        
        except Exception:
            # Fallback to empty schema if extraction fails
            pass
        
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