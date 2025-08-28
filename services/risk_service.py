"""
Risk Analysis Service for Maverick UI

Integrates the PortfolioRiskAnalyzer with portfolio graphs and factor returns
to provide comprehensive risk analysis for the Maverick application.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime

# Add paths for spark module imports
sys.path.append('/Users/rafet/Workspace/Spark')
sys.path.append('/Users/rafet/Workspace/Spark/spark-portfolio')

logger = logging.getLogger(__name__)


class MaverickRiskService:
    """
    Risk analysis service for Maverick UI integrating PortfolioRiskAnalyzer.
    
    This service provides a bridge between the Maverick UI data structures
    and the Spark portfolio risk analysis framework.
    """
    
    def __init__(self):
        """Initialize the risk service."""
        self.portfolio_graph = None
        self.factor_returns = None
        self.risk_analyzer = None
        self.current_analysis_results = None
        self.analysis_cache = {}
        
    def set_portfolio_graph(self, portfolio_graph):
        """
        Set the portfolio graph for analysis.
        
        Args:
            portfolio_graph: PortfolioGraph instance from spark-portfolio
        """
        self.portfolio_graph = portfolio_graph
        self.risk_analyzer = None  # Reset analyzer when portfolio changes
        self.current_analysis_results = None
        self.analysis_cache.clear()
        
        if portfolio_graph:
            try:
                from spark.portfolio.risk_analyzer import PortfolioRiskAnalyzer
                self.risk_analyzer = PortfolioRiskAnalyzer(portfolio_graph)
                logger.info(f"Risk analyzer created for portfolio with {len(portfolio_graph.components)} components")
            except ImportError as e:
                logger.warning(f"Could not import PortfolioRiskAnalyzer: {e}")
                self.risk_analyzer = None
    
    def set_factor_returns(self, factor_returns: pd.DataFrame):
        """
        Set factor returns data for analysis.
        
        Args:
            factor_returns: DataFrame with factors as columns, dates as index
        """
        self.factor_returns = factor_returns
        # Clear cache when factor returns change
        self.analysis_cache.clear()
        self.current_analysis_results = None
        
        if factor_returns is not None:
            logger.info(f"Factor returns set: {factor_returns.shape[1]} factors, {factor_returns.shape[0]} periods")
    
    def is_ready_for_analysis(self) -> bool:
        """Check if service is ready to perform risk analysis."""
        return (
            self.portfolio_graph is not None and 
            self.factor_returns is not None and 
            self.risk_analyzer is not None
        )
    
    def run_risk_analysis(self, 
                         root_component_id: str = None,
                         force_refresh: bool = False,
                         include_time_series: bool = True,
                         populate_hierarchical: bool = False,
                         **kwargs) -> Dict[str, Any]:
        """
        Run comprehensive risk analysis using PortfolioRiskAnalyzer.
        
        Args:
            root_component_id: Root component for analysis (default: first component)
            force_refresh: Force fresh analysis ignoring cache
            include_time_series: Include comprehensive time series data
            populate_hierarchical: Enable bulk hierarchical population of component data
            **kwargs: Additional arguments for risk analysis
            
        Returns:
            Dictionary with analysis results and metadata
        """
        if not self.is_ready_for_analysis():
            return self._create_error_result("Risk service not ready for analysis")
        
        # Determine root component
        if root_component_id is None:
            root_component_id = list(self.portfolio_graph.components.keys())[0]
        
        # Check cache (include hierarchical flag in cache key)
        cache_key = f"{root_component_id}_{include_time_series}_{populate_hierarchical}_{hash(str(kwargs))}"
        if not force_refresh and cache_key in self.analysis_cache:
            logger.info(f"Returning cached analysis for {root_component_id}")
            return self.analysis_cache[cache_key]
        
        try:
            logger.info(f"Running risk analysis for component '{root_component_id}'")
            
            if populate_hierarchical:
                # NEW: Use enhanced risk analysis with hierarchical population
                logger.info("Running analysis with hierarchical population enabled")
                
                # Run factor risk decomposition with visitor
                visitor = self.risk_analyzer.decompose_factor_risk(
                    root_component_id=root_component_id,
                    factor_returns=self.factor_returns
                )
                
                # Create schema using factory with hierarchical population
                try:
                    from spark.risk.schema_factory import RiskSchemaFactory
                    risk_schema = RiskSchemaFactory.from_visitor_results(
                        visitor=visitor,
                        component_id=root_component_id,
                        analysis_type='hierarchical',
                        populate_all_components=True  # Enable bulk hierarchical storage
                    )
                    # Log schema creation success (visitor traversal details not directly accessible)
                    processed_components = len(visitor._processed_components) if hasattr(visitor, '_processed_components') else 'unknown'
                    logger.info(f"Hierarchical schema created successfully, processed {processed_components} components")
                except ImportError as e:
                    logger.warning(f"Could not use hierarchical schema factory: {e}")
                    # Fallback to standard analysis
                    risk_schema = self.risk_analyzer.get_riskresult(
                        root_component_id=root_component_id,
                        factor_returns=self.factor_returns,
                        include_time_series=include_time_series,
                        **kwargs
                    )
            else:
                # Standard risk analysis using PortfolioRiskAnalyzer
                risk_schema = self.risk_analyzer.get_riskresult(
                    root_component_id=root_component_id,
                    factor_returns=self.factor_returns,
                    include_time_series=include_time_series,
                    **kwargs
                )
            
            # Convert schema to Maverick-friendly format
            analysis_result = self._convert_schema_to_maverick_format(
                risk_schema, 
                root_component_id,
                include_time_series
            )
            
            # Add metadata
            analysis_result['metadata'] = {
                'analysis_type': 'hierarchical_factor_risk' if populate_hierarchical else 'standard_factor_risk',
                'root_component': root_component_id,
                'factor_count': self.factor_returns.shape[1],
                'component_count': len(self.portfolio_graph.components),
                'analysis_timestamp': datetime.now().isoformat(),
                'include_time_series': include_time_series,
                'populate_hierarchical': populate_hierarchical,
                'schema_type': str(type(risk_schema).__name__),
                'hierarchical_components': len(risk_schema.get_all_component_risk_results()) if populate_hierarchical and hasattr(risk_schema, 'get_all_component_risk_results') else 0
            }
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis_result
            self.current_analysis_results = analysis_result
            
            logger.info(f"Risk analysis completed successfully for {root_component_id}")
            return analysis_result
            
        except Exception as e:
            error_msg = f"Risk analysis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_result(error_msg)
    
    def _convert_schema_to_maverick_format(self, 
                                          risk_schema,
                                          root_component_id: str,
                                          include_time_series: bool) -> Dict[str, Any]:
        """
        SIMPLIFIED: Direct schema delegation - no conversion needed.
        
        Args:
            risk_schema: RiskResultSchema from PortfolioRiskAnalyzer
            root_component_id: Root component ID
            include_time_series: Whether time series data was included
            
        Returns:
            Dictionary with schema directly embedded
        """
        return {
            'success': True,
            'root_component': root_component_id,
            'schema': risk_schema,  # Direct schema access - single source of truth
            'ui_data': risk_schema.to_ui_format(root_component_id) if hasattr(risk_schema, 'to_ui_format') else {}
        }
    
    def get_component_risk_analysis(self, component_id: str, lens: str = 'portfolio') -> Dict[str, Any]:
        """
        SIMPLIFIED: Get risk analysis using schema delegation.
        
        Args:
            component_id: Component identifier
            lens: Analysis lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Component-specific risk analysis from schema
        """
        if not self.current_analysis_results:
            return self._create_error_result("No analysis results available")
        
        # Get schema directly from results
        schema = self.current_analysis_results.get('schema')
        if not schema:
            return self._create_error_result("No schema available")
        
        try:
            # Use schema methods directly - single source of truth
            return {
                'success': True,
                'component_id': component_id,
                'lens': lens,
                'data': {
                    'core_metrics': schema.get_ui_metrics(component_id, lens),
                    'contributions': {
                        'by_factor': schema.get_ui_contributions(component_id, lens, 'by_factor'),
                        'by_asset': schema.get_ui_contributions(component_id, lens, 'by_asset'),
                        'by_component': schema.get_ui_contributions(component_id, lens, 'by_component')
                    },
                    'exposures': {
                        'factor_exposures': schema.get_ui_exposures(component_id, lens)
                    },
                    'weights': schema.get_ui_weights(component_id),
                    'matrices': schema.get_ui_matrices(component_id, lens)
                }
            }
            
        except Exception as e:
            return self._create_error_result(f"Failed to get component analysis: {str(e)}")
    
    def get_factor_analysis(self) -> Dict[str, Any]:
        """Get factor-level analysis results."""
        if not self.current_analysis_results:
            return self._create_error_result("No analysis results available")
        
        return {
            'success': True,
            'factor_analysis': self.current_analysis_results.get('factor_analysis', {}),
            'factors': list(self.factor_returns.columns) if self.factor_returns is not None else []
        }
    
    def get_time_series_data(self) -> Dict[str, Any]:
        """Get time series data from current analysis."""
        if not self.current_analysis_results:
            return self._create_error_result("No analysis results available")
        
        return {
            'success': True,
            'time_series': self.current_analysis_results.get('time_series', {}),
            'metadata': self.current_analysis_results.get('metadata', {})
        }
    
    def get_hierarchy_structure(self) -> Dict[str, Any]:
        """Get portfolio hierarchy structure from current analysis."""
        if not self.current_analysis_results:
            return self._create_error_result("No analysis results available")
        
        return {
            'success': True,
            'hierarchy': self.current_analysis_results.get('hierarchy', {}),
            'components': list(self.portfolio_graph.components.keys()) if self.portfolio_graph else []
        }
    
    def get_available_lenses(self, component_id: str) -> List[str]:
        """Get available analysis lenses for a component."""
        if not self.current_analysis_results:
            return []
        
        component_analysis = self.current_analysis_results.get('component_analysis', {})
        if component_id in component_analysis:
            return list(component_analysis[component_id].keys())
        
        return ['portfolio', 'benchmark', 'active']  # Default lenses
    
    def validate_analysis_results(self) -> Dict[str, Any]:
        """Validate the current analysis results."""
        if not self.current_analysis_results:
            return {'valid': False, 'message': 'No analysis results available'}
        
        validation = self.current_analysis_results.get('validation', {})
        
        return {
            'valid': validation.get('passes', False),
            'checks': validation.get('checks', {}),
            'details': validation.get('details', {}),
            'summary': 'Analysis validation completed'
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
        self.current_analysis_results = None
        logger.info("Risk analysis cache cleared")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            'portfolio_graph_loaded': self.portfolio_graph is not None,
            'factor_returns_loaded': self.factor_returns is not None,
            'risk_analyzer_available': self.risk_analyzer is not None,
            'ready_for_analysis': self.is_ready_for_analysis(),
            'current_analysis_available': self.current_analysis_results is not None,
            'cache_size': len(self.analysis_cache),
            'portfolio_components': len(self.portfolio_graph.components) if self.portfolio_graph else 0,
            'factor_count': self.factor_returns.shape[1] if self.factor_returns is not None else 0
        }


# Factory function for easy instantiation
def create_risk_service() -> MaverickRiskService:
    """Create a new Maverick risk service instance."""
    return MaverickRiskService()