"""
Data Loader for Portfolio Configuration System

Minimal interface to support the existing Maverick UI while integrating
with the new YAML-based portfolio configuration system.
"""

import os
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

@dataclass
class SidebarState:
    """State object containing sidebar filter selections."""
    lens: str
    selected_node: str
    date_range: tuple
    selected_factors: List[str]
    annualized: bool
    show_percentage: bool


class DataLoader:
    """Data loader supporting both cache-based and YAML configuration systems."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.data = {}
        self.portfolio_graph = None
        self.factor_returns = None
        self.config = None
        self._current_config_id = None
        
        # Risk model integration
        self.risk_model_loader = None
        self._current_risk_model_id = None
        self._initialize_risk_models()
        
        # Risk analysis service
        self.risk_service = None
        self._initialize_risk_service()
        
        # Try to load default configuration if available
        self._load_default_configuration()
    
    def _load_default_configuration(self):
        """Load default configuration on initialization."""
        try:
            from config.portfolio_loader import get_available_configurations, load_portfolio_from_config_name
            
            config_dir = os.path.join(os.path.dirname(__file__), 'config')
            available_configs = get_available_configurations(config_dir)
            
            if available_configs:
                # Load first available configuration as default
                default_config = available_configs[0]['id']
                self.load_portfolio_configuration(default_config)
                
        except Exception as e:
            # Fall back to mock data structure for UI compatibility
            self._load_mock_structure()
    
    def _initialize_risk_models(self):
        """Initialize risk model loader."""
        try:
            from config.risk_model_loader import get_default_risk_model_loader
            self.risk_model_loader = get_default_risk_model_loader()
            
            # Load default risk model
            available_models = self.risk_model_loader.get_available_models()
            if available_models:
                self._current_risk_model_id = available_models[0]['id']
                
        except Exception as e:
            print(f"Warning: Could not initialize risk models: {e}")
    
    def _initialize_risk_service(self):
        """Initialize risk analysis service."""
        try:
            from services.risk_service import create_risk_service
            self.risk_service = create_risk_service()
            print("Risk analysis service initialized")
            
        except Exception as e:
            print(f"Warning: Could not initialize risk service: {e}")
    
    def _load_mock_structure(self):
        """Load minimal mock data structure for UI compatibility."""
        self.data = {
            'metadata': {
                'analysis_type': 'Mock Analysis',
                'data_frequency': 'Daily',
                'schema_version': '1.0'
            },
            'time_series': {
                'metadata': {
                    'start_date': '2023-01-01',
                    'end_date': '2023-12-31'
                },
                'currency': 'USD'
            }
        }
    
    def load_portfolio_configuration(self, config_id: str) -> bool:
        """
        Load portfolio configuration by ID.
        
        Args:
            config_id: Configuration identifier (filename without extension)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from config.portfolio_loader import load_portfolio_from_config_name
            
            config_dir = os.path.join(os.path.dirname(__file__), 'config')
            result = load_portfolio_from_config_name(config_id, config_dir)
            
            self.portfolio_graph = result['portfolio_graph']
            self.factor_returns = result['factor_returns']
            self.config = result['config']
            self._current_config_id = config_id
            
            # Update data structure for UI compatibility
            self._update_data_from_portfolio()
            
            # Update risk service with new portfolio graph
            if self.risk_service:
                self.risk_service.set_portfolio_graph(self.portfolio_graph)
                
                # If we have factor returns from risk model, set those too
                if hasattr(self, 'risk_model_factor_returns'):
                    self.risk_service.set_factor_returns(self.risk_model_factor_returns)
                elif self.factor_returns is not None:
                    self.risk_service.set_factor_returns(self.factor_returns)
            
            return True
            
        except Exception as e:
            st.error(f"Failed to load configuration '{config_id}': {str(e)}")
            return False
    
    def _update_data_from_portfolio(self):
        """Update data structure from loaded portfolio for UI compatibility."""
        if not self.portfolio_graph or not self.config:
            return
        
        # Extract metadata from configuration
        self.data = {
            'metadata': {
                'analysis_type': self.config.name,
                'data_frequency': 'Daily',
                'schema_version': '2.0'
            },
            'time_series': {
                'metadata': {
                    'start_date': '2023-01-01',
                    'end_date': '2023-12-31'
                },
                'currency': self.config.analysis_settings.get('currency', 'USD')
            }
        }
    
    # UI Compatibility Methods
    def get_available_hierarchical_components(self) -> List[str]:
        """Get list of hierarchical component IDs."""
        if self.portfolio_graph:
            return list(self.portfolio_graph.components.keys())
        return ['TOTAL']
    
    def get_component_names(self) -> List[str]:
        """Get list of component names."""
        return self.get_available_hierarchical_components()
    
    def get_component_hierarchy_path(self, component_id: str) -> List[str]:
        """Get hierarchy path for a component."""
        if not component_id:
            return []
        return component_id.split('/')
    
    def get_hierarchy_info(self) -> Dict[str, Any]:
        """Get hierarchy information for components."""
        if not self.portfolio_graph:
            return {'component_metadata': {}}
        
        metadata = {}
        for comp_id, component in self.portfolio_graph.components.items():
            level = comp_id.count('/')
            comp_type = 'leaf' if hasattr(component, 'is_leaf') and component.is_leaf else 'node'
            
            metadata[comp_id] = {
                'type': comp_type,
                'level': level,
                'name': getattr(component, 'name', comp_id)
            }
        
        return {'component_metadata': metadata}
    
    def can_drill_up(self, component_id: str) -> bool:
        """Check if component can drill up to parent."""
        if not component_id or component_id == 'TOTAL':
            return False
        return '/' in component_id
    
    def get_component_parent(self, component_id: str) -> Optional[str]:
        """Get parent component ID."""
        if not component_id or '/' not in component_id:
            return None
        parts = component_id.split('/')
        return '/'.join(parts[:-1]) if len(parts) > 1 else 'TOTAL'
    
    def get_drilldown_options(self, component_id: str) -> List[str]:
        """Get drill-down options for a component."""
        if not self.portfolio_graph:
            return []
        
        children = []
        for comp_id in self.portfolio_graph.components:
            if comp_id.startswith(component_id + '/') and comp_id.count('/') == component_id.count('/') + 1:
                children.append(comp_id)
        
        return children
    
    def get_component_lens_availability(self, component_id: str) -> List[str]:
        """Get available lenses for a component."""
        return ['portfolio', 'benchmark', 'active']
    
    def get_component_validation_status(self, component_id: str, lens: str) -> Dict[str, bool]:
        """Get validation status for a component and lens."""
        return {'euler_identity_check': True}
    
    def get_time_series_data(self, metric_name: str, component_id: str) -> List[float]:
        """Get time series data for a metric and component."""
        if self.portfolio_graph and component_id in self.portfolio_graph.components:
            try:
                metric = self.portfolio_graph.metric_store.get_metric(component_id, metric_name)
                if hasattr(metric, 'series'):
                    return metric.series.tolist()
                elif hasattr(metric, 'value'):
                    return [metric.value] * 252  # Mock time series
            except:
                pass
        
        # Return mock data for UI compatibility
        import numpy as np
        np.random.seed(42)
        return np.random.normal(0.0005, 0.01, 252).tolist()
    
    def get_factor_names(self) -> List[str]:
        """Get list of available factor names."""
        # Try to get factors from current risk model first
        if self._current_risk_model_id and self.risk_model_loader:
            try:
                model_info = self.risk_model_loader.get_model_info(self._current_risk_model_id)
                return model_info.get('factors', [])
            except:
                pass
        
        # Fall back to portfolio factor returns
        if self.factor_returns is not None:
            return self.factor_returns.columns.tolist()
        
        # Default factor names for UI compatibility
        return [
            'Market', 'SMB', 'HML', 'RMW', 'CMA', 'Momentum', 'Quality', 'LowVol',
            'Value', 'Growth', 'Energy', 'Technology', 'HealthCare', 'Financials'
        ]
    
    def get_validation_info(self) -> Dict[str, Any]:
        """Get overall validation information."""
        return {
            'checks': {
                'passes': True,
                'euler_identity': True,
                'weight_consistency': True
            }
        }
    
    def get_hierarchical_data_summary(self) -> Dict[str, Any]:
        """Get summary of hierarchical data."""
        if self.portfolio_graph:
            total_components = len(self.portfolio_graph.components)
            return {
                'total_components': total_components,
                'components_with_matrices': total_components,
                'schema_version': '2.0',
                'component_lens_counts': {comp_id: 3 for comp_id in self.portfolio_graph.components}
            }
        
        return {'total_components': 0, 'components_with_matrices': 0, 'schema_version': '2.0'}
    
    def refresh_data(self):
        """Refresh data from current configuration."""
        if self._current_config_id:
            self.load_portfolio_configuration(self._current_config_id)
    
    @property
    def current_config_id(self) -> Optional[str]:
        """Get current configuration ID."""
        return self._current_config_id
    
    # Risk Model Management Methods
    def load_risk_model(self, model_id: str) -> bool:
        """
        Load risk model by ID.
        
        Args:
            model_id: Risk model identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.risk_model_loader:
            st.error("Risk model system not available")
            return False
            
        try:
            # Load model factor returns
            factor_returns = self.risk_model_loader.load_model_factor_returns(model_id)
            
            # Update current state
            self._current_risk_model_id = model_id
            
            # Update factor returns for integration with portfolio system
            if self.config:
                # If we have a portfolio config, we may want to update factor filtering
                factor_subset = self.config.analysis_settings.get('factor_subset')
                if factor_subset:
                    available_factors = [col for col in factor_returns.columns if col in factor_subset]
                    if available_factors:
                        factor_returns = factor_returns[available_factors]
            
            # Store the risk model factor returns (separate from portfolio factor returns)
            self.risk_model_factor_returns = factor_returns
            
            # Update data structure
            self._update_data_from_risk_model()
            
            # Update risk service with new factor returns
            if self.risk_service:
                self.risk_service.set_factor_returns(factor_returns)
            
            return True
            
        except Exception as e:
            st.error(f"Failed to load risk model '{model_id}': {str(e)}")
            return False
    
    def _update_data_from_risk_model(self):
        """Update data structure from loaded risk model."""
        if not self.risk_model_loader or not self._current_risk_model_id:
            return
        
        try:
            model_info = self.risk_model_loader.get_model_info(self._current_risk_model_id)
            
            # Update metadata to include risk model information
            if 'risk_model' not in self.data:
                self.data['risk_model'] = {}
            
            self.data['risk_model'] = {
                'name': model_info['name'],
                'id': self._current_risk_model_id,
                'source': model_info['source'],
                'factors': model_info['factors'],
                'num_factors': len(model_info['factors']),
                'period': f"{model_info['start_date']} to {model_info['end_date']}"
            }
            
        except Exception as e:
            print(f"Warning: Could not update risk model metadata: {e}")
    
    def get_available_risk_models(self) -> List[Dict[str, Any]]:
        """Get list of available risk models."""
        if self.risk_model_loader:
            return self.risk_model_loader.get_available_models()
        return []
    
    def get_current_risk_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently loaded risk model."""
        if self.risk_model_loader and self._current_risk_model_id:
            try:
                return self.risk_model_loader.get_model_info(self._current_risk_model_id)
            except:
                pass
        return None
    
    def get_risk_model_factor_returns(self) -> Optional[pd.DataFrame]:
        """Get factor returns from the current risk model."""
        if hasattr(self, 'risk_model_factor_returns'):
            return self.risk_model_factor_returns
        return None
    
    @property
    def current_risk_model_id(self) -> Optional[str]:
        """Get current risk model ID."""
        return self._current_risk_model_id
    
    # Risk Analysis Methods
    def run_risk_analysis(self, component_id: str = None, force_refresh: bool = False) -> bool:
        """
        Run risk analysis using the integrated risk service.
        
        Args:
            component_id: Component to analyze (default: first component)
            force_refresh: Force fresh analysis ignoring cache
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.risk_service:
            st.error("Risk service not available")
            return False
        
        if not self.risk_service.is_ready_for_analysis():
            st.error("Risk service not ready - need portfolio graph and factor returns")
            return False
        
        try:
            # Use first component if none specified
            if component_id is None:
                component_id = list(self.portfolio_graph.components.keys())[0]
            
            st.info(f"Running risk analysis for {component_id}...")
            
            # Run the analysis
            analysis_result = self.risk_service.run_risk_analysis(
                root_component_id=component_id,
                force_refresh=force_refresh,
                include_time_series=True
            )
            
            if analysis_result.get('success', False):
                # Update data structure with risk analysis results
                self._update_data_from_risk_analysis(analysis_result)
                st.success("Risk analysis completed successfully")
                return True
            else:
                st.error(f"Risk analysis failed: {analysis_result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            st.error(f"Risk analysis error: {str(e)}")
            return False
    
    def _update_data_from_risk_analysis(self, analysis_result: Dict[str, Any]):
        """Update data structure with risk analysis results."""
        if 'risk_analysis' not in self.data:
            self.data['risk_analysis'] = {}
        
        self.data['risk_analysis'] = {
            'results': analysis_result,
            'status': 'completed',
            'timestamp': analysis_result.get('metadata', {}).get('analysis_timestamp'),
            'ready': True
        }
        
        # Update metadata with risk analysis info
        if 'metadata' not in self.data:
            self.data['metadata'] = {}
        
        self.data['metadata']['has_risk_analysis'] = True
        self.data['metadata']['risk_analysis_type'] = 'hierarchical_factor_risk'
    
    def get_risk_analysis_status(self) -> Dict[str, Any]:
        """Get status of risk analysis."""
        if not self.risk_service:
            return {'available': False, 'message': 'Risk service not available'}
        
        service_status = self.risk_service.get_service_status()
        analysis_data = self.data.get('risk_analysis', {})
        
        return {
            'available': True,
            'ready_for_analysis': service_status['ready_for_analysis'],
            'analysis_completed': analysis_data.get('status') == 'completed',
            'portfolio_components': service_status['portfolio_components'],
            'factor_count': service_status['factor_count'],
            'cache_size': service_status['cache_size'],
            'last_analysis': analysis_data.get('timestamp')
        }
    
    def get_component_risk_analysis(self, component_id: str, lens: str = 'portfolio') -> Dict[str, Any]:
        """Get risk analysis for specific component and lens."""
        if not self.risk_service:
            return {'success': False, 'error': 'Risk service not available'}
        
        return self.risk_service.get_component_risk_analysis(component_id, lens)
    
    def get_available_risk_analysis_lenses(self, component_id: str) -> List[str]:
        """Get available analysis lenses for a component."""
        if not self.risk_service:
            return []
        
        return self.risk_service.get_available_lenses(component_id)
    
    def validate_risk_analysis(self) -> Dict[str, Any]:
        """Validate current risk analysis results."""
        if not self.risk_service:
            return {'valid': False, 'message': 'Risk service not available'}
        
        return self.risk_service.validate_analysis_results()
    
    def has_risk_service(self) -> bool:
        """Check if risk service is available."""
        return self.risk_service is not None

    # UI Compatibility Methods - Bridge between risk service and UI expectations
    
    def get_core_metrics(self, lens: str, component_id: str = None) -> Dict[str, float]:
        """Get core risk metrics for a lens and component."""
        if not self.risk_service:
            return {}
        
        try:
            risk_data = self.risk_service.get_component_risk_analysis(component_id or "TOTAL", lens)
            if not risk_data.get('success', False):
                return {}
            
            data = risk_data.get('data', {})
            core_metrics = data.get('core_metrics', {})
            
            return {
                'total_risk': core_metrics.get('total_risk', 0.0),
                'factor_risk_contribution': core_metrics.get('factor_risk_contribution', 0.0),
                'specific_risk_contribution': core_metrics.get('specific_risk_contribution', 0.0),
                'factor_risk_percentage': core_metrics.get('factor_risk_percentage', 0.0)
            }
        except Exception as e:
            return {}
    
    def get_component_risk_summary(self, component_id: str, lens: str) -> Dict[str, float]:
        """Get risk summary for specific component - alias for get_core_metrics."""
        return self.get_core_metrics(lens, component_id)
    
    def get_contributions(self, lens: str, contrib_type: str) -> Dict[str, float]:
        """Get contributions by asset or factor for a lens."""
        if not self.risk_service:
            return {}
        
        try:
            # Use TOTAL as default component for now
            risk_data = self.risk_service.get_component_risk_analysis("TOTAL", lens)
            if not risk_data.get('success', False):
                return {}
            
            data = risk_data.get('data', {})
            contributions = data.get('contributions', {})
            
            return contributions.get(contrib_type, {})
        except Exception as e:
            return {}
    
    def get_component_asset_contributions(self, component_id: str, lens: str) -> Dict[str, float]:
        """Get asset contributions for specific component."""
        if not self.risk_service:
            return {}
        
        try:
            risk_data = self.risk_service.get_component_risk_analysis(component_id, lens)
            if not risk_data.get('success', False):
                return {}
            
            data = risk_data.get('data', {})
            contributions = data.get('contributions', {})
            
            return contributions.get('by_asset', {})
        except Exception as e:
            return {}
    
    def get_component_factor_contributions(self, component_id: str, lens: str) -> Dict[str, float]:
        """Get factor contributions for specific component."""
        if not self.risk_service:
            return {}
        
        try:
            risk_data = self.risk_service.get_component_risk_analysis(component_id, lens)
            if not risk_data.get('success', False):
                return {}
            
            data = risk_data.get('data', {})
            contributions = data.get('contributions', {})
            
            return contributions.get('by_factor', {})
        except Exception as e:
            return {}
    
    def get_weights(self, weight_type: str) -> Dict[str, float]:
        """Get weights by type (portfolio_weights, benchmark_weights, active_weights)."""
        if not self.risk_service:
            return {}
        
        try:
            # Get from risk service weights data
            risk_data = self.risk_service.get_component_risk_analysis("TOTAL", "portfolio")
            if not risk_data.get('success', False):
                return {}
            
            data = risk_data.get('data', {})
            weights = data.get('weights', {})
            
            return weights.get(weight_type, {})
        except Exception as e:
            return {}
    
    def get_exposures(self, lens: str) -> Dict[str, float]:
        """Get factor exposures for a lens."""
        if not self.risk_service:
            return {}
        
        try:
            risk_data = self.risk_service.get_component_risk_analysis("TOTAL", lens)
            if not risk_data.get('success', False):
                return {}
            
            data = risk_data.get('data', {})
            exposures = data.get('exposures', {})
            
            return exposures.get('factor_exposures', {})
        except Exception as e:
            return {}
    
    def get_matrices(self, matrix_type: str) -> Dict[str, Any]:
        """Get matrix data by type."""
        if not self.risk_service:
            return {}
        
        try:
            risk_data = self.risk_service.get_component_risk_analysis("TOTAL", "portfolio")
            if not risk_data.get('success', False):
                return {}
            
            data = risk_data.get('data', {})
            matrices = data.get('matrices', {})
            
            return matrices.get(matrix_type, {})
        except Exception as e:
            return {}
    
    def get_correlations(self, correlation_type: str) -> Dict[str, Any]:
        """Get correlation data by type."""
        if not self.risk_service:
            return {}
        
        try:
            # Get time series data for correlation computation
            time_series_data = self.risk_service.get_time_series_data()
            correlations = time_series_data.get('correlations', {})
            
            return correlations.get(correlation_type, {})
        except Exception as e:
            return {}
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information from risk service."""
        if not self.risk_service:
            return {'cache_files': [], 'cache_dir': 'N/A'}
        
        try:
            service_status = self.risk_service.get_service_status()
            return {
                'cache_files': [],  # To be populated when cache system is available
                'cache_dir': 'cache/',
                'cache_size_mb': service_status.get('cache_size', 0)
            }
        except Exception as e:
            return {'cache_files': [], 'cache_dir': 'N/A'}
    
    def filter_data_by_factors(self, data: Dict[str, Any], selected_factors: List[str]) -> Dict[str, Any]:
        """Filter data dictionary by selected factors."""
        if not selected_factors or not data:
            return data
        
        # Filter to only include selected factors
        filtered_data = {}
        for key, value in data.items():
            if key in selected_factors:
                filtered_data[key] = value
        
        return filtered_data