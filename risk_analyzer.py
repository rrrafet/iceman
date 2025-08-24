"""
Portfolio Risk Analyzer
=======================

Risk analysis service for hierarchical portfolios using the visitor pattern.
Provides a clean interface for factor risk decomposition while keeping
PortfolioGraph as a pure container.

This module supports:
- Hierarchical portfolio analysis using FactorRiskDecompositionVisitor
- Risk summary extraction and standardization
- Seamless integration with decision attribution framework
"""

from typing import Dict, Optional, Any, TYPE_CHECKING, List
import pandas as pd
import logging

if TYPE_CHECKING:
    from .graph import PortfolioGraph
    from .visitors import FactorRiskDecompositionVisitor
    from spark.risk.estimator import LinearRiskModelEstimator
    from spark.risk.schema import RiskResultSchema
    from spark.risk.strategies import RiskAnalysisStrategy

logger = logging.getLogger(__name__)


class PortfolioRiskAnalyzer:
    """
    Risk analyzer for hierarchical portfolios using the visitor pattern.
    
    Provides a clean interface for factor risk decomposition while keeping
    PortfolioGraph as a pure container. This class serves as a service layer
    between the portfolio structure and risk analysis functionality.
    
    Parameters
    ----------
    portfolio_graph : PortfolioGraph
        Portfolio graph containing the hierarchical structure
    
    Examples
    --------
    # Basic usage
    >>> analyzer = PortfolioRiskAnalyzer(portfolio_graph)
    >>> visitor = analyzer.decompose_factor_risk('portfolio', factor_returns)
    
    # Get standardized risk summary
    >>> summary = analyzer.get_risk_summary('portfolio', factor_returns)
    >>> print(f"Total risk: {summary['portfolio_volatility']:.4f}")
    """
    
    def __init__(self, 
                 portfolio_graph: 'PortfolioGraph',
                 strategy: Optional['RiskAnalysisStrategy'] = None,
                 estimator: Optional['LinearRiskModelEstimator'] = None):
        self.portfolio_graph = portfolio_graph
        self.strategy = strategy  # Will be set to default if None when needed
        self.estimator = estimator  # Will be set to default if None when needed
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    def analysis_type(self) -> str:
        """Get the analysis type being used"""
        return 'hierarchical'
    
    def decompose_factor_risk(
        self,
        root_component_id: str,
        factor_returns: pd.DataFrame,
        estimator: Optional['LinearRiskModelEstimator'] = None,
        portfolio_returns_metric: str = 'portfolio_return',
        benchmark_returns_metric: str = 'benchmark_return',
        portfolio_weight_metric: str = 'portfolio_weight',
        benchmark_weight_metric: str = 'benchmark_weight',
        **kwargs
    ) -> 'FactorRiskDecompositionVisitor':
        """
        Perform hierarchical factor risk decomposition using visitor pattern.
        
        This method creates and runs a FactorRiskDecompositionVisitor to estimate
        factor exposures via OLS regression and decompose risk across the portfolio
        hierarchy.
        
        Parameters
        ----------
        root_component_id : str
            ID of the root component to start decomposition from
        factor_returns : pd.DataFrame
            Factor returns data with factors as columns and dates as index
        estimator : LinearRiskModelEstimator, optional
            Risk model estimator. If None, uses default configuration
        portfolio_returns_metric : str, default 'portfolio_return'
            Name of portfolio returns metric in metric store
        benchmark_returns_metric : str, default 'benchmark_return'
            Name of benchmark returns metric in metric store
        portfolio_weight_metric : str, default 'portfolio_weight'
            Name of portfolio weight metric in metric store
        benchmark_weight_metric : str, default 'benchmark_weight'
            Name of benchmark weight metric in metric store
        **kwargs
            Additional arguments passed to the visitor
            
        Returns
        -------
        FactorRiskDecompositionVisitor
            Visitor instance with completed risk decomposition results
            
        Raises
        ------
        ValueError
            If root component not found or required parameters missing
        """
        self.logger.info(f"Performing hierarchical factor risk decomposition for '{root_component_id}'")
        
        if root_component_id not in self.portfolio_graph.components:
            raise ValueError(f"Component '{root_component_id}' not found in portfolio graph")
        
        from .visitors import FactorRiskDecompositionVisitor
        
        # Create weight service for optimized weight calculations
        weight_service = self.portfolio_graph.create_weight_service()
        
        # Use injected estimator if available, otherwise use provided or default
        final_estimator = estimator or self.estimator
        if final_estimator is None:
            # Import default estimator here to avoid circular imports
            from spark.risk.estimator import LinearRiskModelEstimator
            final_estimator = LinearRiskModelEstimator()
            
        visitor = FactorRiskDecompositionVisitor(
            factor_returns=factor_returns,
            estimator=final_estimator,
            metric_store=self.portfolio_graph.metric_store,
            portfolio_returns_metric=portfolio_returns_metric,
            benchmark_returns_metric=benchmark_returns_metric,
            portfolio_weight_metric=portfolio_weight_metric,
            benchmark_weight_metric=benchmark_weight_metric,
            weight_service=weight_service,
            **kwargs,
        )
        
        root_component = self.portfolio_graph.components[root_component_id]
        root_component.accept(visitor)
        
        self.logger.info(f"Completed hierarchical risk decomposition for component '{root_component_id}'")
        return visitor
    
    
    def get_risk_summary(
        self,
        root_component_id: str,
        factor_returns: pd.DataFrame,
        estimator: Optional['LinearRiskModelEstimator'] = None,
        include_time_series: bool = False,
        **kwargs
    ) -> 'RiskResultSchema':
        """
        Get a standardized risk summary for hierarchical portfolio analysis.
        
        Returns a unified schema with key risk metrics extracted from the
        FactorRiskDecompositionVisitor results. Can optionally include
        comprehensive time series and hierarchy data.
        
        Parameters
        ----------
        root_component_id : str
            Component ID for analysis
        factor_returns : pd.DataFrame
            Factor returns data
        estimator : LinearRiskModelEstimator, optional
            Risk model estimator
        include_time_series : bool, default False
            If True, creates comprehensive schema with time series and hierarchy data
            If False, creates basic schema with risk decomposition only (backward compatible)
        **kwargs
            Additional arguments
            
        Returns
        -------
        RiskResultSchema
            Standardized risk summary in unified schema format
        """
        if include_time_series:
            # Use comprehensive approach
            return self.get_comprehensive_schema(
                root_component_id=root_component_id,
                factor_returns=factor_returns,
                estimator=estimator,
                **kwargs
            )
        else:
            # Use original approach for backward compatibility
            visitor = self.decompose_factor_risk(
                root_component_id=root_component_id,
                factor_returns=factor_returns, 
                estimator=estimator,
                **kwargs
            )
            
            return self._extract_hierarchical_schema(visitor)
    
    def _extract_hierarchical_schema(self, visitor: 'FactorRiskDecompositionVisitor') -> 'RiskResultSchema':
        """
        Extract standardized schema from hierarchical visitor results.
        
        Converts the hierarchical visitor results into unified schema format.
        """
        from spark.risk.schema import RiskResultSchema, AnalysisType
        
        try:
            # Get basic metrics from visitor
            total_risk = getattr(visitor, 'total_active_risk', 0.0)
            factor_risk = getattr(visitor, 'factor_risk_contribution', 0.0)
            specific_risk = getattr(visitor, 'specific_risk_contribution', 0.0)
            
            # Get names
            factor_names = list(getattr(visitor, 'factor_names', []))
            component_names = list(getattr(visitor, 'component_results', {}).keys())
            
            # Create schema
            schema = RiskResultSchema(
                analysis_type=AnalysisType.HIERARCHICAL,
                asset_names=component_names,  # For hierarchical, assets are components
                factor_names=factor_names,
                component_ids=component_names
            )
            
            # Set core metrics
            schema.set_core_metrics(
                total_risk=total_risk,
                factor_risk_contribution=factor_risk,
                specific_risk_contribution=specific_risk
            )
            
            # Try to extract detailed contributions if available
            try:
                component_results = getattr(visitor, 'component_results', {})
                if component_results:
                    # Extract component contributions
                    component_contrib = {}
                    for comp_name, result in component_results.items():
                        if hasattr(result, 'total_risk') or 'total_risk' in result:
                            risk_val = getattr(result, 'total_risk', result.get('total_risk', 0.0))
                            component_contrib[comp_name] = risk_val
                    
                    if component_contrib:
                        schema.set_asset_contributions(component_contrib)
                
                # Try to extract factor exposures/contributions if available
                if hasattr(visitor, 'factor_exposures'):
                    factor_exposures = getattr(visitor, 'factor_exposures', {})
                    if factor_exposures:
                        schema.set_factor_exposures(factor_exposures)
                
                if hasattr(visitor, 'factor_contributions'):
                    factor_contrib = getattr(visitor, 'factor_contributions', {})
                    if factor_contrib:
                        schema.set_factor_contributions(factor_contrib)
                        
            except Exception as extraction_error:
                self.logger.debug(f"Could not extract detailed contributions: {extraction_error}")
            
            # Add context information
            schema.add_context_info('visitor_type', type(visitor).__name__)
            schema.add_context_info('analysis_method', 'hierarchical_visitor')
            schema.add_context_info('component_count', len(component_names))
            
            # Add validation (basic check)
            validation_results = {
                'extraction_successful': True,
                'visitor_type': type(visitor).__name__,
                'components_analyzed': len(component_names)
            }
            schema.set_validation_results(validation_results)
            
            return schema
            
        except Exception as e:
            self.logger.warning(f"Failed to extract hierarchical schema: {e}")
            
            # Return minimal schema if extraction fails
            schema = RiskResultSchema(
                analysis_type=AnalysisType.HIERARCHICAL,
                asset_names=[],
                factor_names=[]
            )
            
            schema.set_core_metrics(0.0, 0.0, 0.0)
            schema.set_validation_results({
                'extraction_successful': False,
                'error': str(e)
            })
            
            return schema
    
    def _extract_portfolio_returns(self) -> Dict[str, pd.Series]:
        """Extract portfolio returns for all components from metric store."""
        portfolio_returns = {}
        metric_store = self.portfolio_graph.metric_store
        
        for component_id in self.portfolio_graph.components.keys():
            # Try different metric names for portfolio returns
            for metric_name in ['portfolio_return', 'port_ret', 'returns']:
                metric = metric_store.get_metric(component_id, metric_name)
                if metric and hasattr(metric, 'value'):
                    returns_data = metric.value()
                    if isinstance(returns_data, pd.Series):
                        portfolio_returns[component_id] = returns_data
                        break
                    elif hasattr(returns_data, 'values') and len(returns_data.values) > 0:
                        # Create series from array data if available
                        portfolio_returns[component_id] = pd.Series(returns_data.values)
                        break
                        
        return portfolio_returns
    
    def _extract_benchmark_returns(self) -> Dict[str, pd.Series]:
        """Extract benchmark returns for all components from metric store."""
        benchmark_returns = {}
        metric_store = self.portfolio_graph.metric_store
        
        for component_id in self.portfolio_graph.components.keys():
            # Try different metric names for benchmark returns
            for metric_name in ['benchmark_return', 'bench_ret', 'benchmark_returns']:
                metric = metric_store.get_metric(component_id, metric_name)
                if metric and hasattr(metric, 'value'):
                    returns_data = metric.value()
                    if isinstance(returns_data, pd.Series):
                        benchmark_returns[component_id] = returns_data
                        break
                    elif hasattr(returns_data, 'values') and len(returns_data.values) > 0:
                        # Create series from array data if available
                        benchmark_returns[component_id] = pd.Series(returns_data.values)
                        break
                        
        return benchmark_returns
    
    def _extract_component_relationships(self) -> Dict[str, Dict[str, Any]]:
        """Extract component parent-child relationships from portfolio graph."""
        relationships = {}
        adjacency_list = self.portfolio_graph.adjacency_list
        
        for comp_id in self.portfolio_graph.components.keys():
            # Find parent
            parent = None
            for parent_id, children in adjacency_list.items():
                if comp_id in children:
                    parent = parent_id
                    break
            
            # Get children
            children = adjacency_list.get(comp_id, [])
            
            relationships[comp_id] = {
                "parent": parent,
                "children": children
            }
            
        return relationships
    
    def _extract_component_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Extract component metadata from portfolio graph."""
        metadata = {}
        
        for comp_id, component in self.portfolio_graph.components.items():
            metadata[comp_id] = {
                "component_id": comp_id,
                "type": "leaf" if component.is_leaf() else "node",
                "level": len(comp_id.split('/')) - 1 if '/' in comp_id else 0,
                "path": comp_id
            }
            
        return metadata
    
    def get_comprehensive_schema(
        self,
        root_component_id: str,
        factor_returns: pd.DataFrame,
        estimator: Optional['LinearRiskModelEstimator'] = None,
        **kwargs
    ) -> 'RiskResultSchema':
        """
        Create comprehensive risk schema with all available data.
        
        This method combines data from all sources available to the analyzer:
        - Risk decomposition results from visitor
        - Time series data from portfolio graph metric store
        - Hierarchy structure from portfolio graph
        - Factor returns from input
        
        Parameters
        ----------
        root_component_id : str
            Component ID for analysis root
        factor_returns : pd.DataFrame
            Factor returns data
        estimator : LinearRiskModelEstimator, optional
            Risk model estimator override
        **kwargs
            Additional arguments passed to decomposition
            
        Returns
        -------
        RiskResultSchema
            Comprehensive schema with risk results, time series, and hierarchy data
        """
        self.logger.info(f"Creating comprehensive schema for '{root_component_id}'")
        
        # 1. Run factor risk decomposition
        visitor = self.decompose_factor_risk(
            root_component_id=root_component_id,
            factor_returns=factor_returns,
            estimator=estimator,
            **kwargs
        )
        
        # 2. Extract time series data from portfolio graph
        portfolio_returns = self._extract_portfolio_returns()
        benchmark_returns = self._extract_benchmark_returns()
        
        # 3. Convert factor returns to dictionary
        factor_returns_dict = {
            factor_name: factor_returns[factor_name] 
            for factor_name in factor_returns.columns
        }
        
        # 4. Create schema using decoupled factory method
        from spark.risk.schema_factory import RiskSchemaFactory
        
        schema = RiskSchemaFactory.from_visitor_results_with_time_series(
            visitor=visitor,
            component_id=root_component_id,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            factor_returns=factor_returns_dict,
            dates=factor_returns.index.tolist(),
            analysis_type="hierarchical"
        )
        
        # 5. Add hierarchy data from portfolio graph
        try:
            component_relationships = self._extract_component_relationships()
            component_metadata = self._extract_component_metadata()
            
            schema.set_hierarchy_structure(
                root_component=root_component_id,
                component_relationships=component_relationships,
                component_metadata=component_metadata,
                adjacency_list=self.portfolio_graph.adjacency_list
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to set hierarchy structure: {e}")
            schema.add_context_info('hierarchy_extraction_error', str(e))
        
        # 6. Add analyzer context information
        schema.add_context_info('analyzer_type', self.__class__.__name__)
        schema.add_context_info('portfolio_components_count', len(self.portfolio_graph.components))
        schema.add_context_info('time_series_portfolio_components', len(portfolio_returns))
        schema.add_context_info('time_series_benchmark_components', len(benchmark_returns))
        schema.add_context_info('extraction_method', 'comprehensive_analyzer')
        
        self.logger.info(f"Comprehensive schema created with {len(portfolio_returns)} portfolio and {len(benchmark_returns)} benchmark time series")
        
        return schema
    
    def __repr__(self) -> str:
        """String representation of the analyzer"""
        return f"PortfolioRiskAnalyzer(components={len(self.portfolio_graph.components)})"


# Convenience factory function  
def create_portfolio_risk_analyzer(portfolio_graph: 'PortfolioGraph') -> PortfolioRiskAnalyzer:
    """
    Create risk analyzer for hierarchical portfolio.
    
    This is a convenience factory function that simply wraps the constructor
    for consistency with other factory patterns in the codebase.
    
    Parameters
    ----------
    portfolio_graph : PortfolioGraph
        Portfolio graph to analyze
        
    Returns
    -------
    PortfolioRiskAnalyzer
        Analyzer configured for hierarchical analysis
    """
    return PortfolioRiskAnalyzer(portfolio_graph)