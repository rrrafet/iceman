"""
Risk Analysis Service for Maverick UI

Simplified risk analysis service that uses direct mapping from visitor metric store
to provide clean, decoupled risk analysis for the Maverick application.
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
    Simplified risk analysis service for Maverick UI.
    
    This service uses direct mapping from visitor metric store data to provide
    clean, decoupled access to risk analysis results without complex conversions.
    All data is extracted directly from the visitor's metric store after 
    decompose_factor_risk() runs.
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
                         **kwargs) -> Dict[str, Any]:
        """
        Run risk analysis using direct mapping from visitor metric store.
        
        Args:
            root_component_id: Root component for analysis (default: first component)
            force_refresh: Force fresh analysis ignoring cache
            **kwargs: Additional arguments for risk analysis
            
        Returns:
            Dictionary with analysis results and metadata
        """
        if not self.is_ready_for_analysis():
            return self._create_error_result("Risk service not ready for analysis")
        
        # Determine root component
        if root_component_id is None:
            root_component_id = list(self.portfolio_graph.components.keys())[0]
        
        # Check cache
        cache_key = f"{root_component_id}_{hash(str(kwargs))}"
        if not force_refresh and cache_key in self.analysis_cache:
            logger.info(f"Returning cached analysis for {root_component_id}")
            return self.analysis_cache[cache_key]
        
        try:
            logger.info(f"Running risk analysis for component '{root_component_id}'")
            
            # Run factor risk decomposition using visitor pattern
            visitor = self.risk_analyzer.decompose_factor_risk(
                root_component_id=root_component_id,
                factor_returns=self.factor_returns,
                **kwargs
            )
            
            # Use direct mapping factory to create schema from visitor metric store
            from spark.risk.schema_factory import RiskSchemaFactory
            schema = RiskSchemaFactory.from_visitor_direct_mapping(
                visitor=visitor,
                root_component_id=root_component_id,
                map_full_hierarchy=True
            )
            
            # Create simplified result structure
            analysis_result = {
                'success': True,
                'root_component': root_component_id,
                'schema': schema,
                'visitor': visitor,  # Keep visitor reference for direct access
                'metadata': {
                    'analysis_type': 'direct_mapping',
                    'root_component': root_component_id,
                    'factor_count': self.factor_returns.shape[1],
                    'component_count': len(self.portfolio_graph.components),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'extraction_method': 'visitor_metric_store'
                }
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
    
    
    def get_component_risk_analysis(self, component_id: str, lens: str = 'portfolio') -> Dict[str, Any]:
        """
        Get component risk analysis directly from schema hierarchical data.
        
        Args:
            component_id: Component identifier
            lens: Analysis lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Component-specific risk analysis
        """
        if not self.current_analysis_results:
            return self._create_error_result("No analysis results available")
        
        schema = self.current_analysis_results.get('schema')
        if not schema:
            return self._create_error_result("No schema available")
        
        try:
            # Get hierarchical data directly from schema
            hierarchical_data = schema.get_hierarchical_risk_data()
            component_data = hierarchical_data.get(component_id, {})
            lens_data = component_data.get(lens, {})
            
            return {
                'success': True,
                'component_id': component_id,
                'lens': lens,
                'data': lens_data
            }
            
        except Exception as e:
            return self._create_error_result(f"Failed to get component analysis: {str(e)}")
    
    def get_factor_analysis(self) -> Dict[str, Any]:
        """Get factor-level analysis results."""
        if not self.current_analysis_results:
            return self._create_error_result("No analysis results available")
        
        schema = self.current_analysis_results.get('schema')
        factor_names = schema.factor_names if schema else []
        
        # Extract factor contributions from root component
        factor_contributions = {}
        if schema:
            hierarchical_data = schema.get_hierarchical_risk_data()
            root_component = self.current_analysis_results['metadata']['root_component']
            root_data = hierarchical_data.get(root_component, {})
            portfolio_data = root_data.get('portfolio', {})
            factor_contributions = portfolio_data.get('factor_contributions', {})
        
        return {
            'success': True,
            'factor_analysis': {
                'contributions': factor_contributions
            },
            'factors': factor_names
        }
    
    def get_time_series_data(self) -> Dict[str, Any]:
        """Get time series data from current analysis."""
        if not self.current_analysis_results:
            return self._create_error_result("No analysis results available")
        
        schema = self.current_analysis_results.get('schema')
        time_series_data = {}
        
        if schema and hasattr(schema, 'get_time_series_data'):
            time_series_data = schema.get_time_series_data()
        
        return {
            'success': True,
            'time_series': time_series_data,
            'metadata': self.current_analysis_results.get('metadata', {})
        }
    
    def get_hierarchy_structure(self) -> Dict[str, Any]:
        """Get portfolio hierarchy structure from current analysis."""
        if not self.current_analysis_results:
            return self._create_error_result("No analysis results available")
        
        schema = self.current_analysis_results.get('schema')
        hierarchy_data = {}
        
        if schema:
            hierarchical_data = schema.get_hierarchical_risk_data()
            hierarchy_data = {
                'components': list(hierarchical_data.keys()),
                'component_count': len(hierarchical_data)
            }
        
        return {
            'success': True,
            'hierarchy': hierarchy_data,
            'components': list(self.portfolio_graph.components.keys()) if self.portfolio_graph else []
        }
    
    def get_available_lenses(self, component_id: str) -> List[str]:
        """Get available analysis lenses for a component."""
        if not self.current_analysis_results:
            return []
        
        schema = self.current_analysis_results.get('schema')
        if schema:
            hierarchical_data = schema.get_hierarchical_risk_data()
            component_data = hierarchical_data.get(component_id, {})
            return list(component_data.keys())
        
        return ['portfolio', 'benchmark', 'active']  # Default lenses
    
    def validate_analysis_results(self) -> Dict[str, Any]:
        """Validate the current analysis results."""
        if not self.current_analysis_results:
            return {'valid': False, 'message': 'No analysis results available'}
        
        schema = self.current_analysis_results.get('schema')
        validation_results = {'valid': True, 'message': 'Analysis available'}
        
        if schema and hasattr(schema, 'get_validation_results'):
            validation_results = schema.get_validation_results()
        
        return {
            'valid': validation_results.get('valid', True),
            'checks': validation_results,
            'summary': 'Direct mapping analysis validation completed'
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