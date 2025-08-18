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

from typing import Dict, Optional, Any, TYPE_CHECKING
import pandas as pd
import logging

if TYPE_CHECKING:
    from .graph import PortfolioGraph
    from .visitors import FactorRiskDecompositionVisitor
    from ..risk.estimator import LinearRiskModelEstimator
    from ..risk.schema import RiskResultSchema

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
    
    def __init__(self, portfolio_graph: 'PortfolioGraph'):
        self.portfolio_graph = portfolio_graph
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
        
        visitor = FactorRiskDecompositionVisitor(
            factor_returns=factor_returns,
            estimator=estimator,
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
        **kwargs
    ) -> 'RiskResultSchema':
        """
        Get a standardized risk summary for hierarchical portfolio analysis.
        
        Returns a unified schema with key risk metrics extracted from the
        FactorRiskDecompositionVisitor results.
        
        Parameters
        ----------
        root_component_id : str
            Component ID for analysis
        factor_returns : pd.DataFrame
            Factor returns data
        estimator : LinearRiskModelEstimator, optional
            Risk model estimator
        **kwargs
            Additional arguments
            
        Returns
        -------
        RiskResultSchema
            Standardized risk summary in unified schema format
        """
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
        from ..risk.schema import RiskResultSchema, AnalysisType
        
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