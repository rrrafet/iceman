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
                         **kwargs) -> Dict[str, Any]:
        """
        Run comprehensive risk analysis using PortfolioRiskAnalyzer.
        
        Args:
            root_component_id: Root component for analysis (default: first component)
            force_refresh: Force fresh analysis ignoring cache
            include_time_series: Include comprehensive time series data
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
        cache_key = f"{root_component_id}_{include_time_series}_{hash(str(kwargs))}"
        if not force_refresh and cache_key in self.analysis_cache:
            logger.info(f"Returning cached analysis for {root_component_id}")
            return self.analysis_cache[cache_key]
        
        try:
            logger.info(f"Running risk analysis for component '{root_component_id}'")
            
            # Run the risk analysis using PortfolioRiskAnalyzer
            risk_schema = self.risk_analyzer.get_risk_summary(
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
                'analysis_type': 'hierarchical_factor_risk',
                'root_component': root_component_id,
                'factor_count': self.factor_returns.shape[1],
                'component_count': len(self.portfolio_graph.components),
                'analysis_timestamp': datetime.now().isoformat(),
                'include_time_series': include_time_series,
                'schema_type': str(type(risk_schema).__name__)
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
        Convert RiskResultSchema to Maverick UI format.
        
        Args:
            risk_schema: RiskResultSchema from PortfolioRiskAnalyzer
            root_component_id: Root component ID
            include_time_series: Whether time series data was included
            
        Returns:
            Dictionary in Maverick UI format
        """
        try:
            result = {
                'success': True,
                'root_component': root_component_id,
                'risk_decomposition': {},
                'hierarchy': {},
                'validation': {},
                'time_series': {},
                'factor_analysis': {}
            }
            
            # Extract core risk metrics
            if hasattr(risk_schema, 'core_metrics'):
                core_metrics = risk_schema.core_metrics
                result['risk_decomposition'] = {
                    'total_risk': getattr(core_metrics, 'total_risk', 0.0),
                    'factor_risk_contribution': getattr(core_metrics, 'factor_risk_contribution', 0.0),
                    'specific_risk_contribution': getattr(core_metrics, 'specific_risk_contribution', 0.0),
                    'factor_risk_percentage': (
                        getattr(core_metrics, 'factor_risk_contribution', 0.0) / 
                        max(getattr(core_metrics, 'total_risk', 1e-10), 1e-10) * 100
                    ),
                    'specific_risk_percentage': (
                        getattr(core_metrics, 'specific_risk_contribution', 0.0) / 
                        max(getattr(core_metrics, 'total_risk', 1e-10), 1e-10) * 100
                    )
                }
            
            # Extract factor contributions
            if hasattr(risk_schema, 'factor_contributions'):
                result['factor_analysis']['contributions'] = dict(risk_schema.factor_contributions)
            
            # Extract asset contributions
            if hasattr(risk_schema, 'asset_contributions'):
                result['factor_analysis']['asset_contributions'] = dict(risk_schema.asset_contributions)
            
            # Extract hierarchy data if available
            if hasattr(risk_schema, 'hierarchy') and risk_schema.hierarchy:
                hierarchy_data = risk_schema.hierarchy
                result['hierarchy'] = {
                    'root_component': getattr(hierarchy_data, 'root_component', root_component_id),
                    'component_metadata': getattr(hierarchy_data, 'component_metadata', {}),
                    'adjacency_list': getattr(hierarchy_data, 'adjacency_list', {}),
                    'component_relationships': getattr(hierarchy_data, 'component_relationships', {})
                }
            
            # Extract time series data if available
            if include_time_series and hasattr(risk_schema, 'time_series'):
                time_series_data = risk_schema.time_series
                
                result['time_series'] = {
                    'dates': getattr(time_series_data, 'dates', []),
                    'portfolio_returns': getattr(time_series_data, 'portfolio_returns', {}),
                    'benchmark_returns': getattr(time_series_data, 'benchmark_returns', {}),
                    'factor_returns': getattr(time_series_data, 'factor_returns', {}),
                    'metadata': {
                        'start_date': getattr(time_series_data, 'dates', [None])[0] if getattr(time_series_data, 'dates', []) else None,
                        'end_date': getattr(time_series_data, 'dates', [None, None])[-1] if getattr(time_series_data, 'dates', []) else None,
                        'frequency': 'B',  # Business daily
                        'currency': 'USD'
                    }
                }
            
            # Extract component-level analysis if available
            if hasattr(risk_schema, 'data') and hasattr(risk_schema.data, 'component_risk_analysis'):
                component_analysis = risk_schema.data.component_risk_analysis
                
                result['component_analysis'] = {}
                for component_id, lenses in component_analysis.items():
                    result['component_analysis'][component_id] = {}
                    
                    for lens, analysis_data in lenses.items():
                        result['component_analysis'][component_id][lens] = {
                            'total_risk': analysis_data.get('total_risk', 0.0),
                            'factor_risk_contribution': analysis_data.get('factor_risk_contribution', 0.0),
                            'specific_risk_contribution': analysis_data.get('specific_risk_contribution', 0.0),
                            'factor_contributions': analysis_data.get('factor_contributions', {}),
                            'asset_contributions': analysis_data.get('asset_contributions', {}),
                            'validation': {
                                'euler_identity_check': analysis_data.get('euler_identity_check', False),
                                'validation_summary': analysis_data.get('validation_summary', '')
                            }
                        }
            
            # Extract validation results
            if hasattr(risk_schema, 'validation'):
                result['validation'] = {
                    'passes': getattr(risk_schema.validation, 'overall_success', True),
                    'checks': getattr(risk_schema.validation, 'validation_results', {}),
                    'details': getattr(risk_schema.validation, 'details', {})
                }
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to convert schema to Maverick format: {e}")
            return self._create_error_result(f"Schema conversion failed: {str(e)}")
    
    def get_component_risk_analysis(self, component_id: str, lens: str = 'portfolio') -> Dict[str, Any]:
        """
        Get risk analysis for a specific component and lens.
        
        Args:
            component_id: Component identifier
            lens: Analysis lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Component-specific risk analysis
        """
        if not self.current_analysis_results:
            return self._create_error_result("No analysis results available")
        
        try:
            component_analysis = self.current_analysis_results.get('component_analysis', {})
            
            if component_id in component_analysis:
                if lens in component_analysis[component_id]:
                    return {
                        'success': True,
                        'component_id': component_id,
                        'lens': lens,
                        'analysis': component_analysis[component_id][lens]
                    }
            
            return self._create_error_result(f"No {lens} analysis found for component {component_id}")
            
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